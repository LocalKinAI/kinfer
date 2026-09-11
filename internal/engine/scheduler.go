package engine

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/sampling"
)

// prefillChunk is how many prompt tokens one slot may contribute to a single
// batch. Bounding it is what keeps an agent arriving with an 8k system prompt
// from stalling every conversation already generating.
const prefillChunk = 512

// A scheduler runs every conversation through one llama.cpp context.
//
// The reason is arithmetic rather than tidiness. Generating a token means
// reading every weight in the model, so a 7B at Q4 moves 4.7 GB per token no
// matter how many conversations are waiting. Put eight of them in the same
// forward pass and the weights are still read once — the GPU does eight times
// the arithmetic on data it was going to fetch anyway. Measured on an M3 Ultra
// with Qwen2.5-0.5B: 300 tok/s alone, 1337 at a batch of 8, 2825 at 32.
//
// Which sizes are worth using is not obvious. Step time nearly doubles between
// a batch of 8 and 12 and then stays flat to 32, so twelve concurrent requests
// produce less total throughput than eight. See the table in README; slot
// counts come from {8, 32, 64, 128} and never from 12-16.
//
// Exactly one goroutine — run — touches the context. Everything else hands it
// work over a channel. A llama.cpp context is mutable state with no locking of
// its own, and two goroutines decoding into it would interleave KV writes.
type scheduler struct {
	lctx  llama.Context
	vocab llama.Vocab
	tpl   *chat.Template

	// ctxPerSeq is how many tokens one slot may hold. llama.cpp divides the
	// context between sequences, so eight slots over 32k leaves 4k each.
	ctxPerSeq int

	batch *llama.BatchBuilder
	slots []*slot

	incoming chan *job
	stop     chan struct{}
	stopped  chan struct{}

	// debug counters, printed when KINFER_DEBUG_SCHED is set. Batch
	// composition is the thing worth watching: if steps mostly carry one token
	// the slots are not filling and the batching is theoretical.
	debug     bool
	steps     int
	tokSum    int
	activeSum int
	lastLog   time.Time
}

// job is one request waiting for, or receiving, a reply.
type job struct {
	ctx    context.Context
	msgs   []chat.Message
	params GenParams

	// frags carries generated text; it is closed when the reply ends. err holds
	// the failure, if any, and is only read after frags is closed.
	frags chan string
	err   error
	done  chan struct{}
}

func (j *job) finish(err error) {
	j.err = err
	close(j.frags)
	close(j.done)
}

// slot is one sequence's worth of state: a seq_id in the shared KV cache, the
// sampler that owns its repetition history, and the text produced so far.
type slot struct {
	seq int32
	job *job

	prompt []llama.Token
	nPast  int32 // tokens of this sequence already in the KV cache
	next   llama.Token

	sampler *sampling.Sampler
	out     strings.Builder
	emitted int
	nGen    int
	maxGen  int

	// logitIdx is where this slot's logits landed in the batch just decoded,
	// or -1 when it contributed no token whose output was requested.
	logitIdx int32
}

func newScheduler(lctx llama.Context, vocab llama.Vocab, tpl *chat.Template, nSlots, nCtx, batchCap int) *scheduler {
	s := &scheduler{
		lctx:      lctx,
		vocab:     vocab,
		tpl:       tpl,
		ctxPerSeq: nCtx / nSlots,
		batch:     llama.NewBatchBuilder(batchCap),
		slots:     make([]*slot, nSlots),
		incoming:  make(chan *job, 64),
		stop:      make(chan struct{}),
		stopped:   make(chan struct{}),
		debug:     os.Getenv("KINFER_DEBUG_SCHED") != "",
	}
	for i := range s.slots {
		s.slots[i] = &slot{seq: int32(i), logitIdx: -1}
	}
	go s.run()
	return s
}

// submit queues a request and returns its job. The caller reads frags until it
// closes, then checks err.
func (s *scheduler) submit(j *job) error {
	select {
	case s.incoming <- j:
		return nil
	case <-s.stop:
		return errors.New("engine is closed")
	}
}

func (s *scheduler) close() {
	close(s.stop)
	<-s.stopped
}

// run is the only goroutine that touches the context.
func (s *scheduler) run() {
	defer close(s.stopped)

	for {
		s.admit()

		if s.active() == 0 {
			// Nothing to do. Block rather than spin — an idle fleet should not
			// keep a core warm.
			select {
			case j := <-s.incoming:
				s.start(j)
			case <-s.stop:
				s.drain()
				return
			}
			continue
		}

		select {
		case <-s.stop:
			s.drain()
			return
		default:
		}

		if err := s.step(); err != nil {
			// A decode failure is not attributable to one slot, so every slot
			// in the batch hears about it rather than hanging.
			for _, sl := range s.slots {
				if sl.job != nil {
					s.release(sl, err)
				}
			}
		}
	}
}

// admit fills free slots from the queue without blocking.
func (s *scheduler) admit() {
	for _, sl := range s.slots {
		if sl.job != nil {
			continue
		}
		select {
		case j := <-s.incoming:
			s.startIn(sl, j)
		default:
			return
		}
	}
}

func (s *scheduler) start(j *job) {
	for _, sl := range s.slots {
		if sl.job == nil {
			s.startIn(sl, j)
			return
		}
	}
}

func (s *scheduler) startIn(sl *slot, j *job) {
	if err := j.ctx.Err(); err != nil {
		j.finish(err)
		return
	}

	prompt := s.tpl.Render(j.msgs)
	tokens, err := llama.Tokenize(s.vocab, prompt, true, true)
	if err != nil {
		j.finish(fmt.Errorf("tokenize: %w", err))
		return
	}
	if len(tokens) >= s.ctxPerSeq {
		j.finish(fmt.Errorf("prompt is %d tokens but each slot holds %d "+
			"(context %d split across %d slots)",
			len(tokens), s.ctxPerSeq, s.ctxPerSeq*len(s.slots), len(s.slots)))
		return
	}

	// The previous occupant's cells are still tagged with this seq_id. Drop
	// them, and only them: clearing the whole cache would take the other slots'
	// conversations with it.
	llama.ForgetSequence(s.lctx, sl.seq, -1, -1)

	maxGen := j.params.MaxTokens
	if maxGen <= 0 {
		maxGen = s.ctxPerSeq - len(tokens)
	}

	sl.job = j
	sl.prompt = tokens
	sl.nPast = 0
	sl.nGen = 0
	sl.maxGen = maxGen
	sl.emitted = 0
	sl.out.Reset()
	sl.logitIdx = -1
	sl.sampler = sampling.New(j.params.Params)
}

func (s *scheduler) active() int {
	n := 0
	for _, sl := range s.slots {
		if sl.job != nil {
			n++
		}
	}
	return n
}

// step builds one batch across every active slot, decodes it once, and samples
// a token for each slot that asked for logits.
func (s *scheduler) step() error {
	s.batch.Reset()

	for _, sl := range s.slots {
		if sl.job == nil {
			continue
		}
		if err := sl.job.ctx.Err(); err != nil {
			s.release(sl, err)
			continue
		}

		if int(sl.nPast) < len(sl.prompt) {
			s.addPrefill(sl)
		} else {
			s.addDecode(sl)
		}
	}

	if s.batch.Len() == 0 {
		return nil
	}
	if s.debug {
		s.steps++
		s.tokSum += s.batch.Len()
		s.activeSum += s.active()
		if time.Since(s.lastLog) > time.Second {
			log.Printf("sched: %4d steps/s   mean batch %5.1f tokens   mean active slots %4.1f",
				s.steps, float64(s.tokSum)/float64(s.steps), float64(s.activeSum)/float64(s.steps))
			// Interval statistics, not cumulative: a mean diluted by an idle
			// minute says nothing about what happens under load.
			s.steps, s.tokSum, s.activeSum = 0, 0, 0
			s.lastLog = time.Now()
		}
	}
	if err := s.batch.Decode(s.lctx); err != nil {
		return err
	}

	for _, sl := range s.slots {
		if sl.job == nil || sl.logitIdx < 0 {
			continue
		}
		s.harvest(sl)
	}
	return nil
}

// addPrefill queues the next chunk of a slot's prompt.
//
// Prompts go in in pieces rather than whole so that one agent arriving with an
// 8k system prompt cannot stall every conversation already generating: its
// chunk shares the batch with their single tokens, and the step stays bounded.
func (s *scheduler) addPrefill(sl *slot) {
	space := s.batch.Cap() - s.batch.Len()
	if space <= 0 {
		return
	}
	remaining := len(sl.prompt) - int(sl.nPast)
	n := min(remaining, min(space, prefillChunk))

	for i := 0; i < n; i++ {
		pos := sl.nPast + int32(i)
		last := int(pos) == len(sl.prompt)-1
		idx := s.batch.Add(sl.prompt[pos], pos, sl.seq, last)
		if idx < 0 {
			n = i
			break
		}
		if last {
			sl.logitIdx = idx
		}
	}
	sl.nPast += int32(n)
}

// addDecode queues the one token a generating slot needs this step.
func (s *scheduler) addDecode(sl *slot) {
	if s.batch.Cap()-s.batch.Len() <= 0 {
		return
	}
	idx := s.batch.Add(sl.next, sl.nPast, sl.seq, true)
	if idx < 0 {
		return
	}
	sl.logitIdx = idx
	sl.nPast++
}

// harvest samples a slot's next token and decides whether it is finished.
func (s *scheduler) harvest(sl *slot) {
	idx := sl.logitIdx
	sl.logitIdx = -1

	tok := sl.sampler.Sample(s.lctx, idx)
	if llama.IsEOG(s.vocab, tok) {
		s.finishSlot(sl)
		return
	}

	piece := llama.TokenToPiece(s.vocab, tok, false)
	if piece == "" {
		s.finishSlot(sl)
		return
	}
	sl.out.WriteString(piece)
	sl.next = tok
	sl.nGen++

	// Some templates leak their stop marker as ordinary text; cut there.
	text, hitStop := s.tpl.TrimStop(sl.out.String())

	// Emit only what has become valid UTF-8. A CJK character spans two or three
	// tokens, and half of one is not printable.
	if len(text) > sl.emitted {
		pending := text[sl.emitted:]
		if utf8.ValidString(pending) {
			// Never block here. A client that has stopped reading would
			// otherwise stall the scheduler and with it every other slot. The
			// text stays in sl.out, so a failed send costs nothing: the next
			// token's harvest offers a larger chunk instead.
			select {
			case sl.job.frags <- pending:
				sl.emitted = len(text)
			default:
			}
		}
	}

	if hitStop || sl.nGen >= sl.maxGen || int(sl.nPast) >= s.ctxPerSeq {
		s.finishSlot(sl)
	}
}

// finishSlot ends a reply normally, flushing whatever was held back.
func (s *scheduler) finishSlot(sl *slot) {
	text, _ := s.tpl.TrimStop(sl.out.String())
	if len(text) > sl.emitted {
		// The buffer is sized to hold a whole reply, so this does not block in
		// practice; the default arm keeps a vanished client from proving that
		// wrong at everyone else's expense.
		select {
		case sl.job.frags <- text[sl.emitted:]:
			sl.emitted = len(text)
		default:
		}
	}
	s.release(sl, nil)
}

// release frees a slot and tells its caller the reply is over.
func (s *scheduler) release(sl *slot, err error) {
	if sl.job == nil {
		return
	}
	llama.ForgetSequence(s.lctx, sl.seq, -1, -1)
	if sl.sampler != nil {
		sl.sampler.Close()
		sl.sampler = nil
	}
	sl.job.finish(err)
	sl.job = nil
	sl.prompt = nil
	sl.logitIdx = -1
	sl.out.Reset()
}

// drain ends every in-flight and queued request on shutdown.
func (s *scheduler) drain() {
	err := errors.New("engine is closing")
	for _, sl := range s.slots {
		s.release(sl, err)
	}
	for {
		select {
		case j := <-s.incoming:
			j.finish(err)
		default:
			return
		}
	}
}
