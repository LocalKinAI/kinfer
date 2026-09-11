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

// minPrefixMatch is the shortest pooled prefix worth adopting. Below it the
// bookkeeping outweighs a prefill llama.cpp would have done in microseconds —
// measured at over 17,000 tokens/s for prompt processing.
const minPrefixMatch = 128

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
// That paragraph is about a dense model and does not carry over to a
// mixture-of-experts one. "The weights are read once" is the claim batching
// rests on, and for MoE it is mostly false: each token in the batch routes to
// its own experts, so a second token in the same pass fetches its own weights
// rather than riding along with the first. Measured here, as the cost of
// doubling the tokens in one decode step — 1.00x would be free, 2.00x would be
// no saving at all:
//
//	tokens      ornith-1.5-35b     qwen3.8-flash-next
//	            (35B.A3B)          (125B.A6B)
//	 1 ->  2     1.15x              1.53x
//	 2 ->  4     1.68x              1.57x
//	 4 ->  8     1.62x              1.54x
//	 8 -> 16     1.60x              1.81x
//
// So the saving is 20-40%, and it shrinks as the batch grows. Batching is still
// worth doing — aggregate throughput on ornith climbs 83 -> 166 -> 231 tok/s at
// 1, 8 and 24 streams — but it climbs because more tokens are in flight, not
// because they are cheap.
//
// The consequence worth writing down is for speculative decoding, which has not
// been built and on this evidence should not be. It works by putting k drafted
// tokens into one pass and verifying them together, and it pays only when that
// fatter pass is nearly free — exactly the property the table says is missing.
// Feeding the measured step times through the usual expectation, (1-a^(k+1))
// over (1-a) tokens of progress per step for a per-token acceptance a, with the
// draft model's own passes charged for:
//
//	                        per stream, baseline -> speculative
//	ornith  N=1  k=3 a=0.9   69.9 -> 102.6 tok/s   +47%
//	ornith  N=8  k=3 a=0.9   22.4 ->  22.4         +-0%
//	ornith  N=8  k=3 a=0.75  22.4 ->  17.8         -21%
//	125B    N=1  k=3 a=0.9   33.7 ->  46.0         +37%
//	125B    N=8  k=3 a=0.9    9.1 ->   8.8          -3%
//	125B    N=8  k=3 a=0.75   9.1 ->   7.0         -23%
//
// Speculation buys a lot for one stream and nothing for eight, and below about
// 90% acceptance it costs, because a rejected draft is wasted expert traffic
// rather than wasted arithmetic. A fleet of agents lives in the bottom rows.
// One person waiting for one answer lives in the top ones.
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
	pool  *prefixPool

	// onFatal is called when llama.cpp's backend fails unrecoverably, so the
	// engine can be marked unusable and replaced rather than kept in service.
	onFatal func()

	incoming chan *job
	stop     chan struct{}
	stopped  chan struct{}

	// rate is how fast requests have actually been finishing, which is the only
	// honest basis for telling a refused caller when to come back.
	rate completionRate

	// debug counters, printed when KINFER_DEBUG_SCHED is set. Batch
	// composition is the thing worth watching: if steps mostly carry one token
	// the slots are not filling and the batching is theoretical.
	debug     bool
	steps     int
	tokSum    int
	activeSum int
	admitted  int
	promptTok int
	reusedTok int
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

	// stats is what was counted and measured while answering. The scheduler
	// fills it in before finish, so it is only read after frags is closed.
	stats Stats
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
	reused int   // tokens of nPast that came from the prefix pool
	next   llama.Token

	// publish marks a slot whose prompt has just finished prefilling and is
	// worth pooling. It is acted on after the decode, so the cells exist.
	publish bool

	sampler *sampling.Sampler
	out     strings.Builder
	emitted int
	nGen    int
	maxGen  int

	// admitted is when this slot took the job and began prefilling; evalStart
	// is when the first token was sampled, which is the instant prefill ended
	// and generation began. The gap between them is prompt evaluation, and
	// everything after evalStart is decoding — the split Ollama reports and
	// every tokens-per-second figure divides by.
	admitted   time.Time
	evalStart  time.Time
	promptEval time.Duration

	// logitIdx is where this slot's logits landed in the batch just decoded,
	// or -1 when it contributed no token whose output was requested.
	logitIdx int32
}

func newScheduler(lctx llama.Context, vocab llama.Vocab, tpl *chat.Template, nSlots, nPrefix, ctxPerSeq, batchCap, maxQueue int) *scheduler {
	s := &scheduler{
		lctx:      lctx,
		vocab:     vocab,
		tpl:       tpl,
		ctxPerSeq: ctxPerSeq,
		batch:     llama.NewBatchBuilder(batchCap),
		slots:     make([]*slot, nSlots),
		incoming:  make(chan *job, maxQueue),
		stop:      make(chan struct{}),
		stopped:   make(chan struct{}),
		debug:     os.Getenv("KINFER_DEBUG_SCHED") != "",
	}
	for i := range s.slots {
		s.slots[i] = &slot{seq: int32(i), logitIdx: -1}
	}
	// Pooled prefixes live in sequence ids above the slots.
	s.pool = newPrefixPool(lctx, int32(nSlots), nPrefix, minPrefixMatch)
	go s.run()
	return s
}

// submit queues a request and returns its job. The caller reads frags until it
// closes, then checks err.
func (s *scheduler) submit(j *job) error {
	select {
	case <-s.stop:
		// Same case as a job drained from the queue: nothing of this request
		// happened, so a caller able to reload can simply run it again.
		return ErrNotStarted
	default:
	}

	select {
	case s.incoming <- j:
		return nil
	case <-s.stop:
		return ErrNotStarted
	default:
		// The queue is full. Refuse, rather than block until something times
		// out: a caller told it is 200 deep can back off, and a caller left
		// holding an open connection cannot tell that from a slow model.
		waiting := len(s.incoming)
		return &BusyError{
			Waiting:    waiting,
			Capacity:   cap(s.incoming),
			RetryAfter: s.rate.wait(waiting),
		}
	}
}

// Waiting is how many requests are queued for a slot.
func (s *scheduler) Waiting() int { return len(s.incoming) }

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

			var de *llama.DecodeError
			if errors.As(err, &de) && de.Fatal() {
				// The context is finished. Serving from it would return this
				// same error to every request from now on, which for a fallback
				// runtime means it is down without saying so.
				log.Printf("llama.cpp backend failed fatally (%v) — retiring this model; "+
					"the next request reloads it", err)
				if s.onFatal != nil {
					s.onFatal()
				}
				s.drain()
				return
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

	prompt := s.tpl.Render(j.msgs, j.params.Tools)
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

	// Adopt whatever of this prompt is already in the cache. An agent's system
	// prompt is identical on every turn it takes, so this is usually most of
	// what it sent.
	reused := 0
	if e, n := s.pool.match(tokens); n > 0 {
		s.pool.adopt(e, sl.seq, n)
		reused = n
	}

	maxGen := j.params.MaxTokens
	if maxGen <= 0 {
		maxGen = s.ctxPerSeq - len(tokens)
	}

	if s.debug {
		s.admitted++
		s.promptTok += len(tokens)
		s.reusedTok += reused
	}

	sl.job = j
	sl.prompt = tokens
	sl.nPast = int32(reused)
	sl.reused = reused
	sl.publish = false
	sl.nGen = 0
	sl.maxGen = maxGen
	sl.emitted = 0
	sl.out.Reset()
	sl.logitIdx = -1
	sl.admitted = time.Now()
	sl.evalStart = time.Time{}
	sl.promptEval = 0
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
			reuse := ""
			if s.admitted > 0 {
				reuse = fmt.Sprintf("   admitted %d, prompt %d tokens of which %d reused (%.0f%%)",
					s.admitted, s.promptTok, s.reusedTok,
					100*float64(s.reusedTok)/float64(s.promptTok))
			}
			log.Printf("sched: %4d steps/s   mean batch %5.1f tokens   mean active slots %4.1f%s",
				s.steps, float64(s.tokSum)/float64(s.steps), float64(s.activeSum)/float64(s.steps), reuse)
			// Interval statistics, not cumulative: a mean diluted by an idle
			// minute says nothing about what happens under load.
			s.steps, s.tokSum, s.activeSum = 0, 0, 0
			s.admitted, s.promptTok, s.reusedTok = 0, 0, 0
			s.lastLog = time.Now()
		}
	}
	if err := s.batch.Decode(s.lctx); err != nil {
		return err
	}

	// Take the time only when something is about to be sampled. A step that
	// carried nothing but prompt chunks has no logits to read and no boundary
	// to record, and making the GPU finish for it would stall a pipeline the
	// next decode simply continues.
	var now time.Time
	if s.harvesting() {
		// Decode queues the forward pass and returns before the GPU is done;
		// llama.cpp waits at the first read of the logits, which is inside
		// Sample. Timing the decode without this charges that wait to the
		// measurement *after* the one it belongs to — a 509-token prefill
		// timed at 5 ms and the single token sampled from it at 36, which is
		// the wrong answer twice. Sampling is about to block on precisely this
		// wait, so asking for it here costs nothing and gives every slot in the
		// batch the same instant rather than the order this loop visits them.
		llama.Synchronize(s.lctx)
		now = time.Now()
	}

	for _, sl := range s.slots {
		if sl.job == nil {
			continue
		}
		if sl.publish {
			sl.publish = false
			s.pool.publish(sl.seq, sl.prompt)
		}
		if sl.logitIdx >= 0 {
			s.harvest(sl, now)
		}
	}
	return nil
}

// harvesting reports whether the batch just decoded produced logits for any
// slot, which is what makes this step worth timing and synchronising.
func (s *scheduler) harvesting() bool {
	for _, sl := range s.slots {
		if sl.job != nil && sl.logitIdx >= 0 {
			return true
		}
	}
	return false
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
			// The whole prompt is now in the cache under this slot's sequence.
			// Pool it once the decode lands.
			sl.publish = true
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
//
// now is when the decode these logits came from finished.
func (s *scheduler) harvest(sl *slot, now time.Time) {
	idx := sl.logitIdx
	sl.logitIdx = -1

	if sl.evalStart.IsZero() {
		// First logits for this slot, so the prompt is fully in the cache and
		// prefill is over. addPrefill asks for logits only on the prompt's last
		// token, and the prefix pool never adopts a whole prompt, so this
		// really is the boundary and not a chunk boundary.
		sl.promptEval = now.Sub(sl.admitted)
		sl.evalStart = now
	}

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

	if hitStop {
		s.finishSlot(sl)
		return
	}
	if sl.nGen >= sl.maxGen || int(sl.nPast) >= s.ctxPerSeq {
		sl.job.stats.Truncated = true
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

	// Every ending comes through here — the model stopping, the budget running
	// out, a cancelled request, a dead backend — so this is the one place the
	// measurements have to be handed over. A request that died during prefill
	// reports its prompt and a zero eval, which is what happened.
	sl.job.stats.PromptTokens = len(sl.prompt)
	sl.job.stats.EvalTokens = sl.nGen
	sl.job.stats.PromptEvalDuration = sl.promptEval
	if !sl.evalStart.IsZero() {
		sl.job.stats.EvalDuration = time.Since(sl.evalStart)
	}

	sl.job.finish(err)
	s.rate.done()
	sl.job = nil
	sl.prompt = nil
	sl.logitIdx = -1
	sl.admitted = time.Time{}
	sl.evalStart = time.Time{}
	sl.out.Reset()
}

// ErrNotStarted means a request was still queued when the engine went down, so
// nothing of it ever reached llama.cpp or the client.
//
// It exists to separate the two halves of a shutdown. A slot that was mid-batch
// has already streamed part of an answer, and re-running it would repeat that
// text; there is nothing to do but fail it. A job still in the queue holds only
// its messages and parameters — no KV state, no emitted tokens, no tie to the
// context that died — and can simply be run again on the replacement.
//
// That distinction is most of the blast radius. When a backend failure retired
// a model mid-load-test, 26 requests failed at once and the majority of them
// had never started. Callers that can reload a model should retry this; callers
// that cannot should report it. Engine cannot reload itself, so it reports.
var ErrNotStarted = errors.New("the engine went down before this request started")

// drain ends every in-flight and queued request on shutdown.
//
// The two loops fail their jobs with different errors on purpose — see
// ErrNotStarted.
func (s *scheduler) drain() {
	closing := errors.New("engine is closing")
	for _, sl := range s.slots {
		s.release(sl, closing)
	}
	for {
		select {
		case j := <-s.incoming:
			j.finish(ErrNotStarted)
		default:
			return
		}
	}
}
