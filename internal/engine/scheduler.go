package engine

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/sampling"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// Prefill is first-come, first-served, not fair-shared.
//
// It used to be shared: every slot with prompt left contributed up to 512
// tokens per step, so four arrivals filled a 2048-token batch four ways and
// climbed together. Measured with eight fleet-shaped requests on ornith-1.5-35b
// (3k to 25k tokens each, 93k in all): every one of them saw its first token
// between 82 and 87 seconds, the 3k prompt no sooner than the 25k one, and all
// eight missed the caller's 60-second line. Prefill throughput is fixed by the
// hardware — about 1000 tokens a second here whatever the batch shape — so the
// only thing scheduling decides is who waits for whom. Fair sharing makes
// everyone wait for the largest.
//
// Now the oldest prompt takes all the space a step has, then the next. The
// same eight requests would see first tokens at roughly 3, 6, 13, 20, 34, 48,
// 71 and 87 seconds — six inside the line instead of none — and a slot that
// finishes early starts generating in the same batches that prefill the rest,
// which is the point of continuous batching. Steps are no longer than before:
// batchCapacity bounds them either way, and generating slots are added first,
// so a big arrival still cannot stall a conversation already under way.

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
// with Qwen2.5-0.5B: 309 tok/s alone, 2329 at a batch of 8, 3494 at 16
// (llama-batched-bench, llama.cpp b11998).
//
// Which sizes are worth using is not obvious. Past a batch of 16 the step time
// more than doubles, and it takes 48 to 64 sequences to pay for it, so 24 or
// 32 concurrent requests produce less total throughput than 16. See the table
// in docs/performance.md; slot counts come from {8, 16, 64, 128} and never from
// 17 to about 48.
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
// The consequence worth writing down is for speculative decoding (draft.go),
// of which a mixture of experts gets one token a step. It works by putting k
// drafted tokens into one pass and verifying them together, and it pays only
// when that fatter pass is nearly free — exactly the property the table says
// is missing. Feeding the measured step times through the usual expectation,
// (1-a^(k+1)) over (1-a) tokens of progress per step for a per-token
// acceptance a, with the draft model's own passes charged for:
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
// One person waiting for one answer lives in the top ones. Measured since, with
// the models' own MTP heads drafting: ornith-1.5-35b fell from 108 tok/s to
// 78-85 at four drafts, accepting one in five, and rose to 118-131 at one,
// accepting two in three or more — and Qwen3.8-27B, which is dense and reads
// every weight for one token as for five, rose from 31.5 to 37-69 at four.
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

	// wholePrefixes is the pool's rule for a model with recurrent state, kept
	// so a rebuilt context's pool follows it too; see prefixPool.exact.
	wholePrefixes bool

	// draft guesses each generating slot's next tokens with the model's MTP
	// head, for one pass to check; nil when the model has none or drafting is
	// off. It is rebuilt with the context. See draft.go.
	draft *drafter

	// flex rebuilds the context in another layout when a prompt needs longer
	// slots, and back when nothing does; nil keeps the layout it loaded with.
	// See flex.go.
	flex *flex

	// profile keeps the counts of the experts generation selects from one
	// context to the next; see experts.go.
	profile expertProfile

	// held is the oldest request not yet in a slot. It has been taken off the
	// queue to be sized, and waits here for a slot or for the layout it needs.
	held *job

	// nSlots, nPrefix and holding mirror len(slots), the pool's entries and
	// held != nil for readers on other goroutines: the slots and the pool are
	// replaced when the layout changes.
	nSlots  atomic.Int32
	nPrefix atomic.Int32
	holding atomic.Bool

	// onFatal is called when llama.cpp's backend fails unrecoverably, so the
	// engine can be marked unusable and replaced rather than kept in service.
	onFatal func()

	// roomHint is set when the load left next to nothing of the GPU, and is
	// added to a fatal failure's error: the likeliest reason for it.
	roomHint string

	// mtmd reads pictures, when the model has a projector. fatal is a failure
	// that ended the context outside a step — reading pictures — which the
	// next step reports as its own.
	mtmd  llama.Mtmd
	fatal error

	incoming chan *job
	stop     chan struct{}
	stopped  chan struct{}

	// queueCap is how many requests may wait for a slot that are still
	// wanted, and waiting counts them. incoming has room for more: for the
	// ones whose callers gave up, which leave it only on reaching its head.
	// Counted with them, a burst of callers timing out and asking again
	// filled it with requests nobody waited for, and the ones still wanted
	// were refused.
	queueCap int
	waiting  atomic.Int64

	// rate is how fast requests have actually been finishing, which is the only
	// honest basis for telling a refused caller when to come back.
	rate completionRate

	// limits bound the two ways a request can occupy the server without
	// finishing: holding a slot, and holding a place in the queue.
	limits deadlines

	// Live counters, atomic because /metrics reads them from an HTTP goroutine
	// while run() owns everything else here.
	busy       atomic.Int64
	promptSeen atomic.Int64
	evalSeen   atomic.Int64
	// reusedSeen is the prompt tokens of promptSeen the prefix pool gave a
	// slot instead of the model reading them.
	reusedSeen atomic.Int64

	// drafted and draftsAccepted count the MTP head's drafts the model read
	// and the ones it agreed with, across every context this scheduler had.
	drafted, draftsAccepted atomic.Int64

	// callOpen, thinkOpen and thinkClose are the tokens <tool_call>, <think>
	// and </think>, or sampling.NoToken where one is more than one; see
	// toolGrammar. noGrammar is KINFER_NO_TOOL_GRAMMAR, which leaves calls
	// unchecked.
	callOpen, thinkOpen, thinkClose llama.Token
	noGrammar                       bool

	// callsHeld counts the replies whose tool calls the grammar took hold of,
	// and redrawn the tokens in them drawn again because it ruled out the
	// first: how often it changed what a model wrote.
	callsHeld, redrawn atomic.Int64

	// debug counters, printed when KINFER_DEBUG_SCHED is set. Batch
	// composition is the thing worth watching: if steps mostly carry one token
	// the slots are not filling and the batching is theoretical.
	debug     bool
	steps     int
	tokSum    int
	activeSum int
	admitted  int
	moved     int // replies compact moved down a slot
	promptTok int
	reusedTok int
	lastLog   time.Time

	// The drafting counters as the last debug line left them.
	loggedDrafts, loggedAccepted int64

	// Where a step's time goes, for the same debug line: queueing the forward
	// pass, waiting for the GPU to finish it, and sampling. What is left of the
	// wall clock is the scheduler itself. Timed only when debug is set.
	tDecode, tWait, tSample time.Duration
}

// job is one request waiting for, or receiving, a reply.
type job struct {
	ctx    context.Context
	msgs   []chat.Message
	params GenParams

	// queued is when submit accepted it, so a job the fleet has given up on can
	// be refused rather than handed a slot.
	queued time.Time

	// full is the whole prompt, rendered and tokenized when the job reached
	// the head of the queue, and want is the most slots it runs whole in —
	// the layout it asks for when flex is on, and the slot count otherwise.
	full []llama.Token
	want int

	// resume is the reply the job carries over (GenParams.Resume), tokenized
	// to follow the prompt.
	resume []llama.Token

	// media is a prompt with pictures, cut into text and picture chunks in
	// place of full; see prepareMedia.
	media *llama.MediaPrompt

	// frags carries generated text; it is closed when the reply ends. err holds
	// the failure, if any, and is only read after frags is closed.
	frags chan string
	err   error
	done  chan struct{}

	// stats is what was counted and measured while answering. The scheduler
	// fills it in before finish, so it is only read after frags is closed.
	stats Stats

	// tail is the end of a reply that would not fit in frags, a client
	// having read too slowly: read after frags closes.
	tail string

	// inQueue is true while the job waits in incoming and counts in the
	// scheduler's waiting; unwatch stops the watch that takes it out of the
	// count the moment its caller gives up.
	inQueue atomic.Bool
	unwatch func() bool
}

func (j *job) finish(err error) {
	j.media.Free()
	j.media = nil
	j.err = err
	close(j.frags)
	close(j.done)
}

// checkpoint is a position to pool a prefilling prompt at; see slot.checkpoints.
type checkpoint struct {
	pos    int32
	shared bool
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

	// drafts are the tokens the MTP head expects after next, read with it in
	// this step's pass; see draft.go.
	drafts []llama.Token

	// media is how many cells a prompt with pictures took, read whole by
	// startMedia; its prompt is then empty, and nPast its last position.
	media int

	// publishLen is how much of the prompt to pool once the decode that just
	// landed is in the cache, or 0 for nothing; publishShared marks it as a
	// system prompt other conversations can adopt. Acted on after the decode,
	// so the cells exist.
	publishLen    int
	publishShared bool

	// checkpoints are the positions, in order, at which prefill ends a batch
	// so the pool can take this sequence at exactly that point, recurrent state
	// included. Only a pool that adopts whole entries — a model with recurrent
	// state — needs them: the end of the system prompt, which another
	// conversation of the same agent opens with, and the end of the
	// conversation so far, which its next turn contains.
	checkpoints []checkpoint

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

	// pre is the token sampled at logitIdx ahead of harvest, when presample
	// sampled every slot of a step at once; hasPre says it is there.
	pre    llama.Token
	hasPre bool
}

// newScheduler takes over lctx, laid out as home: from here on the scheduler
// owns the context, frees and rebuilds it when fx changes the layout, and hands
// whatever context it holds last to the engine to free once it has stopped.
func newScheduler(lctx llama.Context, draft *drafter, home layout, vocab llama.Vocab, tpl *chat.Template, batchCap, maxQueue int, limits deadlines, wholePrefixes bool, fx *flex) *scheduler {
	s := &scheduler{
		vocab:         vocab,
		tpl:           tpl,
		batch:         llama.NewBatchBuilder(batchCap),
		incoming:      make(chan *job, maxQueue*givenUpRoom),
		queueCap:      maxQueue,
		limits:        limits,
		wholePrefixes: wholePrefixes,
		flex:          fx,
		stop:          make(chan struct{}),
		stopped:       make(chan struct{}),
		debug:         os.Getenv("KINFER_DEBUG_SCHED") != "",
		callOpen:      oneToken(vocab, "<tool_call>"),
		thinkOpen:     oneToken(vocab, "<think>"),
		thinkClose:    oneToken(vocab, "</think>"),
		noGrammar:     os.Getenv("KINFER_NO_TOOL_GRAMMAR") != "",
	}
	s.install(lctx, draft, home)
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
	case <-s.stopped:
		// The loop has ended (the backend failed fatally) and nothing reads
		// the queue any more. Queued, the request would wait for ever, and
		// its hold on the engine would keep the model it failed on loaded.
		return ErrNotStarted
	default:
	}

	j.queued = time.Now()
	if n := s.waiting.Load(); n >= int64(s.queueCap) {
		return s.busyError(n)
	}
	j.inQueue.Store(true)
	s.waiting.Add(1)
	if j.ctx != nil {
		// A caller that gives up stops counting at once, not when the
		// request reaches the head of the queue.
		j.unwatch = context.AfterFunc(j.ctx, func() { s.leftQueue(j) })
	}

	select {
	case s.incoming <- j:
	case <-s.stop:
		s.taken(j)
		return ErrNotStarted
	default:
		// Room for the wanted ones, but not beside the given-up ones still
		// in the queue.
		s.taken(j)
		return s.busyError(s.waiting.Load())
	}
	select {
	case <-s.stopped:
		// The loop ended while this was being queued, after its last look at
		// the queue: answer what it left there.
		s.drainQueue()
	default:
	}
	return nil
}

// givenUpRoom is how many times queueCap the queue holds, the rest for the
// requests given up on that have not yet reached its head.
const givenUpRoom = 4

// busyError is the refusal of a request with n waiting. Refused rather than
// left to block until something times out: a caller told it is 200 deep can
// back off, and a caller left holding an open connection cannot tell that
// from a slow model.
func (s *scheduler) busyError(n int64) error {
	return &BusyError{
		Waiting:    int(n),
		Capacity:   s.queueCap,
		RetryAfter: s.rate.wait(int(n)),
	}
}

// leftQueue takes j out of the count of requests waiting: it reached the head
// of the queue, or its caller gave up.
func (s *scheduler) leftQueue(j *job) {
	if j.inQueue.CompareAndSwap(true, false) {
		s.waiting.Add(-1)
	}
}

// taken is j off the queue: out of the count, and no longer watched.
func (s *scheduler) taken(j *job) {
	if j.unwatch != nil {
		j.unwatch()
	}
	s.leftQueue(j)
}

// Waiting is how many requests are queued for a slot, counting the one at the
// head that admission has already taken off the queue to size.
func (s *scheduler) Waiting() int {
	n := int(s.waiting.Load())
	if s.holding.Load() {
		n++
	}
	return n
}

func (s *scheduler) close() {
	close(s.stop)
	<-s.stopped
}

// run is the only goroutine that touches the context.
func (s *scheduler) run() {
	defer s.ended()

	for {
		if !s.admit() {
			// A layout change freed the context and nothing could be built in
			// its place. Everything waiting is told so and can run again on
			// a reloaded model.
			if s.onFatal != nil {
				s.onFatal()
			}
			s.drain()
			return
		}

		s.compact()

		if s.active() == 0 && s.fatal == nil {
			// Nothing to do. Block rather than spin — an idle fleet should not
			// keep a core warm.
			select {
			case j := <-s.incoming:
				s.taken(j)
				if s.prepare(j) {
					s.hold(j)
				}
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
			var de *llama.DecodeError
			fatal := errors.As(err, &de) && de.Fatal()

			// A decode failure is not attributable to one slot, so every slot
			// in the batch hears about it rather than hanging. When it ends
			// the context, each hears that its reply was cut short where it
			// stood, which a reloaded model can go on from.
			failed := err
			if fatal {
				failed = fmt.Errorf("%w: %w", ErrInterrupted, err)
				if s.roomHint != "" {
					failed = fmt.Errorf("%w; %s", failed, s.roomHint)
				}
			}
			for _, sl := range s.slots {
				if sl.job != nil {
					s.release(sl, failed)
				}
			}

			if fatal {
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

// admit fills free slots from the queue, oldest first, without blocking. false
// means a layout change left the scheduler without a context.
//
// Oldest first is kept strictly: the request at the head is placed before
// anything behind it is looked at, so one waiting for the slots to drain holds
// back everything after it. See flex.go for why.
func (s *scheduler) admit() bool {
	for {
		j := s.head()
		if j == nil {
			return true
		}
		switch s.place(j) {
		case placeNow:
			s.hold(nil)
			s.startIn(s.freeSlot(), j)
		case placeReshape:
			if !s.reshape(j) {
				return false
			}
		default:
			return true
		}
	}
}

// head is the oldest request not yet in a slot: the one already held, or the
// next off the queue, sized. nil when nothing is waiting.
func (s *scheduler) head() *job {
	for {
		if j := s.held; j != nil {
			if s.stillWanted(j) {
				return j
			}
			s.hold(nil)
			continue
		}
		select {
		case j := <-s.incoming:
			s.taken(j)
			if s.prepare(j) {
				s.hold(j)
			}
		default:
			return nil
		}
	}
}

func (s *scheduler) hold(j *job) {
	s.held = j
	s.holding.Store(j != nil)
}

// stillWanted answers a request nobody is waiting for any more, and reports
// whether it is still worth a slot.
func (s *scheduler) stillWanted(j *job) bool {
	if err := j.ctx.Err(); err != nil {
		j.finish(err)
		return false
	}
	// Spending a slot on an answer nobody is left to read costs the requests
	// behind it, which are the ones still being waited for.
	if el, expired := s.limits.expiredWaiting(j.queued); expired {
		j.finish(&TimeoutError{After: el, Stage: StageQueued})
		return false
	}
	return true
}

// prepare renders and tokenizes a request as it reaches the head of the queue
// and works out the layout it runs whole in. false when it has been answered
// already: abandoned, expired, or not something the template can render.
func (s *scheduler) prepare(j *job) bool {
	if !s.stillWanted(j) {
		return false
	}
	if showsPictures(j.msgs) {
		if s.mtmd == 0 {
			j.finish(PictureError{ErrNoPictures})
			return false
		}
		return s.prepareMedia(j)
	}
	full, err := s.tokenizer(j)(j.msgs)
	if err == nil {
		j.resume, err = s.resumeTokens(j)
	}
	if err != nil {
		j.finish(fmt.Errorf("tokenize: %w", err))
		return false
	}
	j.full = full
	j.want = len(s.slots)
	if s.flex != nil {
		j.want = s.flex.want(len(full)+len(j.resume), j.params.MaxTokens)
	}
	return true
}

// resumeTokens is the reply j carries over (GenParams.Resume), tokenized to
// follow its prompt: without the think opening startIn writes into the reply
// when the prompt itself opened the block.
func (s *scheduler) resumeTokens(j *job) ([]llama.Token, error) {
	text := j.params.Resume
	if s.tpl.OpensThink(j.params.thinking()) {
		text = strings.TrimPrefix(text, "<think>")
	}
	return llama.Tokenize(s.vocab, text, false, true)
}

// tokenizer renders and tokenizes a conversation the way j asked for it.
func (s *scheduler) tokenizer(j *job) func([]chat.Message) ([]llama.Token, error) {
	render := s.renderer(j)
	return func(msgs []chat.Message) ([]llama.Token, error) {
		return llama.Tokenize(s.vocab, render(msgs, j.params.Tools), true, true)
	}
}

// renderer renders a conversation with the thinking j asked for.
func (s *scheduler) renderer(j *job) func([]chat.Message, []tools.Tool) string {
	th := j.params.thinking()
	return func(msgs []chat.Message, ts []tools.Tool) string {
		return s.tpl.RenderWith(msgs, ts, th)
	}
}

func (s *scheduler) startIn(sl *slot, j *job) {
	if !s.stillWanted(j) {
		return
	}
	if j.media != nil {
		s.startMedia(sl, j)
		return
	}

	render := s.renderer(j)
	tokenize := s.tokenizer(j)
	// fitMessages renders the whole conversation first, and prepare already
	// did exactly that; the cuts after it render afresh.
	full := j.full
	fitTokenize := func(msgs []chat.Message) ([]llama.Token, error) {
		if full != nil {
			t := full
			full = nil
			return t, nil
		}
		return tokenize(msgs)
	}
	// A conversation longer than the slot loses its oldest turns rather than
	// being refused, and one that nearly fills it loses enough to leave room
	// for the reply; see fit.go. A reply carried over takes its room first.
	limit := s.ctxPerSeq - len(j.resume)
	budget := limit - replyRoom(s.ctxPerSeq, j.params.MaxTokens)
	msgs, tokens, dropped, err := fitMessages(j.msgs, budget, limit, fitTokenize, s.messageSize)
	j.full = nil // the slot keeps what it runs; the rest is garbage
	if err != nil {
		var big *PromptTooLongError
		if errors.As(err, &big) {
			big.Slots = len(s.slots)
			if s.flex != nil {
				// It already has the longest slot on offer, so fewer slots
				// is not advice that would help.
				big.Slots = 1
			}
		} else {
			err = fmt.Errorf("tokenize: %w", err)
		}
		j.finish(err)
		return
	}
	if dropped > 0 {
		log.Printf("context: dropped the oldest %d of %d messages so the prompt fits a %d-token conversation "+
			"with room to reply — now %d tokens", dropped, len(j.msgs), s.ctxPerSeq, len(tokens))
	}

	// The previous occupant's cells are still tagged with this seq_id. Drop
	// them, and only them: clearing the whole cache would take the other slots'
	// conversations with it.
	s.forget(sl.seq)

	// Adopt whatever of this prompt is already in the cache. An agent's system
	// prompt is identical on every turn it takes, so this is usually most of
	// what it sent.
	reused := 0
	if e, n := s.pool.match(tokens); n > 0 {
		s.pool.adopt(e, sl.seq, n)
		reused = n
	}

	// A pool that can only adopt whole entries needs this prompt pooled where
	// later ones will repeat it whole: at the end of the conversation so far,
	// for its next turn, and at the end of the system prompt, for the next
	// conversation of the same agent. For the first of those: The rendered prompt ends with
	// the template's opening of the reply — "<|im_start|>assistant\n", and for
	// some templates a think block — which the next turn does not reproduce,
	// because there the assistant's turn is rendered with what it said. So the
	// conversation is rendered again with an assistant turn appended, and the
	// two prompts agree exactly as far as the next turn will.
	//
	// Without this, the rule that keeps a hybrid model correct also stopped an
	// agent's next turn reusing anything: measured on ornith-1.5:9b, 0 of 1029
	// tokens, because the pooled prompt ended in an opening the next prompt
	// renders differently.
	var checkpoints []checkpoint
	if s.pool.wholeOnly() {
		// Where this prompt and a rendering of msgs part — taken back to just
		// after a control token. An ordinary token at the edge is tokenized
		// with whatever follows it, and a later prompt has different text
		// there: the "\n" that ended "<|im_start|>assistant\n" came back in the
		// next turn merged into the "\n\n" before a <tool_call>, one token short
		// of whole. Control tokens are never merged with anything.
		agreeWith := func(msgs []chat.Message) int32 {
			ext, err := llama.Tokenize(s.vocab, render(msgs, j.params.Tools), true, true)
			if err != nil {
				return 0
			}
			n := commonPrefix(tokens, ext)
			for n > 0 && !llama.IsControl(s.vocab, tokens[n-1]) {
				n--
			}
			if n <= reused || n < minPrefixMatch {
				return 0
			}
			return int32(n)
		}
		if len(msgs) > 1 && msgs[0].Role == "system" {
			// Rendered with a different first message, the system prompt is
			// where the two part.
			if pos := agreeWith([]chat.Message{msgs[0], {Role: "user", Content: "x"}}); pos > 0 {
				checkpoints = append(checkpoints, checkpoint{pos: pos, shared: true})
			}
		}
		next := append(append([]chat.Message(nil), msgs...), chat.Message{Role: "assistant", Content: "x"})
		if pos := agreeWith(next); pos > 0 && (len(checkpoints) == 0 || pos > checkpoints[0].pos) {
			checkpoints = append(checkpoints, checkpoint{pos: pos})
		}
	}

	// The reply carried over is prefilled after the prompt, and the model
	// goes on from its end. It comes after the pool's checkpoints, which are
	// in the prompt.
	tokens = append(tokens, j.resume...)

	// A caller that named no budget gets the rest of its slot's context. That
	// is a real bound, so kinfer does not also impose a ceiling on num_predict:
	// what a runaway request costs is time, and the wall clock in deadlines
	// bounds that directly. A token ceiling would be a second knob buying the
	// same safety, and the number it wanted would be invented.
	maxGen := j.params.MaxTokens
	if maxGen <= 0 {
		maxGen = s.ctxPerSeq - len(tokens)
	}

	if s.debug {
		s.admitted++
		s.promptTok += len(tokens)
		s.reusedTok += reused
	}

	s.occupy(sl, j, tokens, reused, checkpoints, maxGen)
}

// occupy gives sl to j, its prompt tokens read as far as reused, to reply in
// up to maxGen tokens.
func (s *scheduler) occupy(sl *slot, j *job, tokens []llama.Token, reused int, checkpoints []checkpoint, maxGen int) {
	sl.job = j
	s.busy.Add(1)
	sl.prompt = tokens
	sl.media = 0
	sl.nPast = int32(reused)
	sl.reused = reused
	sl.publishLen = 0
	sl.publishShared = false
	sl.checkpoints = checkpoints
	sl.nGen = 0
	sl.maxGen = maxGen
	sl.emitted = 0
	sl.out.Reset()
	switch {
	case j.params.Resume != "":
		// The caller has the reply so far. It stays in front of what follows,
		// so a stop marker that straddles the two is still found, and it is
		// not sent again.
		sl.out.WriteString(j.params.Resume)
		sl.emitted = len(j.params.Resume)
	case s.tpl.OpensThink(j.params.thinking()):
		// The prompt opened the think block, so the model writes only its
		// close. Reported from here, the reply reads as every other reasoning
		// model's does, and the splitters downstream need no exception.
		sl.out.WriteString("<think>")
	}
	sl.logitIdx = -1
	sl.hasPre = false
	sl.admitted = time.Now()
	sl.evalStart = time.Time{}
	sl.promptEval = 0
	sl.sampler = sampling.New(j.params.Params, llama.NVocab(s.vocab))
	if g, ok := s.toolGrammar(j); ok {
		if err := sl.sampler.Constrain(g); err != nil {
			log.Printf("tool calls go unchecked: %v", err)
		}
	}
}

// toolGrammar is the grammar that holds a job's tool calls to their form and
// to the schemas it declared, for a model that writes them in the XML form: a
// call left open, prose after it, or an integer that is not one then cannot be
// written. A small model gets these wrong often enough that an agent loses
// turns to them. Names it did not declare are let through; see
// tools.FunctionXMLGrammar.
//
// Only a call is held, from its <tool_call> on; whatever the model writes
// before it is drawn as ever. Inside a think block, the prompt's or the
// model's own, nothing is held, so a call the model only thinks about is not
// taken for one. A resumed reply goes unchecked: what it has written so far is
// not known to the grammar.
func (s *scheduler) toolGrammar(j *job) (sampling.Grammar, bool) {
	p := j.params
	if s.noGrammar || len(p.Tools) == 0 || p.Resume != "" ||
		s.tpl.ToolFormat() != tools.FunctionXML || s.callOpen == sampling.NoToken {
		return sampling.Grammar{}, false
	}
	thinking := s.tpl.OpensThink(p.thinking())
	if thinking && s.thinkClose == sampling.NoToken {
		return sampling.Grammar{}, false
	}
	rules, err := tools.FunctionXMLGrammar(p.Tools)
	if err != nil {
		log.Printf("tool calls go unchecked: %v", err)
		return sampling.Grammar{}, false
	}
	return sampling.Grammar{Vocab: s.vocab, Rules: rules, Trigger: s.callOpen,
		Open: s.thinkOpen, Close: s.thinkClose, Thinking: thinking}, true
}

// oneToken is the token text is, when it is exactly one and reads back as
// text, or sampling.NoToken.
func oneToken(v llama.Vocab, text string) llama.Token {
	toks, err := llama.Tokenize(v, text, false, true)
	if err != nil || len(toks) != 1 || llama.TokenToPiece(v, toks[0], true) != text {
		return sampling.NoToken
	}
	return toks[0]
}

// replyRoom is how much of a slot a fitted prompt leaves for the reply: what
// the caller asked for, up to an eighth of the slot. Agents ask for far more
// than a local context holds — Claude Code sends max_tokens 32000 — and a tool
// call rarely needs more than a few thousand.
func replyRoom(ctxPerSeq, maxTokens int) int {
	room := ctxPerSeq / 8
	if maxTokens > 0 {
		room = min(room, maxTokens)
	}
	return room
}

// messageSize is a message's own tokens, without the template around it. It
// only places the points where fitting may cut a conversation, so it needs to
// be the same every time a message is sized, not exact.
func (s *scheduler) messageSize(m chat.Message) int {
	n := len(m.Content) / 4
	if t, err := llama.Tokenize(s.vocab, m.Content, false, false); err == nil {
		n = len(t)
	}
	for _, c := range m.ToolCalls {
		n += (len(c.Name) + len(c.Arguments)) / 3
	}
	return n
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
	if err := s.fatal; err != nil {
		s.fatal = nil
		return err
	}
	s.batch.Reset()

	var prefilling, generating []*slot
	for _, sl := range s.slots {
		if sl.job == nil {
			continue
		}
		if err := sl.job.ctx.Err(); err != nil {
			s.release(sl, err)
			continue
		}
		if el, expired := s.limits.expiredGenerating(sl.evalStart); expired {
			s.release(sl, &TimeoutError{After: el, Stage: StageGenerating})
			continue
		}

		if int(sl.nPast) < len(sl.prompt) {
			prefilling = append(prefilling, sl)
		} else {
			generating = append(generating, sl)
		}
	}
	if s.draft != nil {
		if each := draftsEach(len(generating)); each > 0 {
			s.draft.propose(generating, func(sl *slot) int { return min(each, s.allowance(sl)) })
		} else {
			// Too many are writing for drafts to pay, and the head would read
			// every token they write for nothing: it stops following them,
			// and follows each slot again from its next request.
			for _, sl := range generating {
				s.draft.untrack(sl.seq)
			}
		}
	}
	// Generating slots go in first: a token each, and its drafts, never
	// displaced by however much prompt is waiting behind them.
	for _, sl := range generating {
		s.addDecode(sl)
	}
	for _, sl := range prefillOrder(prefilling) {
		s.addPrefill(sl)
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
			if d, a := s.drafted.Load()-s.loggedDrafts, s.draftsAccepted.Load()-s.loggedAccepted; d > 0 {
				reuse += fmt.Sprintf("   drafts %d, %.0f%% accepted", d, 100*float64(a)/float64(d))
			}
			s.loggedDrafts, s.loggedAccepted = s.drafted.Load(), s.draftsAccepted.Load()
			if s.moved > 0 {
				reuse += fmt.Sprintf("   moved %d replies down a slot", s.moved)
				s.moved = 0
			}
			ms := func(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) / float64(s.steps) }
			wall := float64(time.Since(s.lastLog)) / float64(time.Millisecond) / float64(s.steps)
			log.Printf("sched: %4d steps/s   mean batch %5.1f tokens   mean active slots %4.1f   "+
				"per step %.2f ms: decode %.2f, gpu wait %.2f, sample %.2f, scheduler %.2f%s",
				s.steps, float64(s.tokSum)/float64(s.steps), float64(s.activeSum)/float64(s.steps),
				wall, ms(s.tDecode), ms(s.tWait), ms(s.tSample), wall-ms(s.tDecode)-ms(s.tWait)-ms(s.tSample), reuse)
			llama.PerfPrint(s.lctx)
			llama.PerfReset(s.lctx)
			// Interval statistics, not cumulative: a mean diluted by an idle
			// minute says nothing about what happens under load.
			s.steps, s.tokSum, s.activeSum = 0, 0, 0
			s.tDecode, s.tWait, s.tSample = 0, 0, 0
			s.admitted, s.promptTok, s.reusedTok = 0, 0, 0
			s.lastLog = time.Now()
		}
	}
	var t0, decoded time.Time
	if s.debug {
		t0 = time.Now()
	}
	if err := s.batch.Decode(s.lctx); err != nil {
		// A pool without cells of its own keeps its prompts in whatever the
		// slots are not using, so a full cache is the pool's to give back. The
		// failed decode changed nothing — llama.cpp finds room for every
		// micro-batch before it runs any — so the same batch simply runs again.
		var de *llama.DecodeError
		if !errors.As(err, &de) || de.Code != 1 || !s.pool.clear() {
			return err
		}
		log.Printf("prefix pool: the cache was full — pooled prompts dropped to make room")
		if err := s.batch.Decode(s.lctx); err != nil {
			return err
		}
	}
	if s.debug {
		decoded = time.Now()
		s.tDecode += decoded.Sub(t0)
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
		if s.debug {
			s.tWait += now.Sub(decoded)
		}
	}
	// The head reads what the model just read before anything is pooled or
	// sampled: a pooled prompt takes the head's cells with it, and a rejected
	// draft takes them back.
	if s.draft != nil {
		s.draft.catchUp(s.batch)
	}
	s.presample()

	for _, sl := range s.slots {
		if sl.job == nil {
			continue
		}
		if sl.publishLen > 0 {
			if sl.publishShared {
				s.pool.publishShared(sl.seq, sl.prompt[:sl.publishLen])
			} else {
				s.pool.publish(sl.seq, sl.prompt[:sl.publishLen])
			}
			sl.publishLen = 0
		}
		if sl.logitIdx >= 0 {
			s.harvest(sl, now)
		}
	}
	return nil
}

// presampleMin is how many slots a step must sample before presample shares
// the work out: under it, starting goroutines costs more than it saves.
const presampleMin = 8

// presample samples the first token of every slot with logits this step, on
// as many threads as there are slots or processors. Sampling reads a row of
// the vocabulary's logits and sorts through it — 152k of them for Qwen — and
// done slot after slot it was a third of each step at 128 slots: 8 ms of 24
// on the MacBook Pro. Each slot has a sampler of its own, and the logits are
// only read once the main thread has synchronised and read the first of them,
// which is when llama.cpp puts its output rows in order; harvest then takes
// the tokens in slot order, as before, and samples a draft's further tokens
// itself.
func (s *scheduler) presample() {
	var todo []*slot
	for _, sl := range s.slots {
		if sl.job != nil && sl.logitIdx >= 0 {
			todo = append(todo, sl)
		}
	}
	if len(todo) < presampleMin {
		return
	}
	var t0 time.Time
	if s.debug {
		t0 = time.Now()
	}
	llama.Logits(s.lctx, todo[0].logitIdx, 1)
	workers := min(len(todo), runtime.GOMAXPROCS(0))
	var next atomic.Int32
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := int(next.Add(1)) - 1; i < len(todo); i = int(next.Add(1)) - 1 {
				sl := todo[i]
				sl.pre, sl.hasPre = sl.sampler.Sample(s.lctx, sl.logitIdx), true
			}
		}()
	}
	wg.Wait()
	if s.debug {
		s.tSample += time.Since(t0)
	}
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

// prefillOrder is who gets the batch first: the slot that has been waiting
// longest. Stable, so two admitted in the same instant keep slot order.
func prefillOrder(slots []*slot) []*slot {
	sort.SliceStable(slots, func(i, j int) bool { return slots[i].admitted.Before(slots[j].admitted) })
	return slots
}

// addPrefill queues as much of a slot's remaining prompt as the batch has room
// for. A prompt larger than a step still goes in pieces — batchCapacity bounds
// the step — but the pieces are consecutive steps of one slot, not one slice of
// each.
func (s *scheduler) addPrefill(sl *slot) {
	space := s.batch.Cap() - s.batch.Len()
	if space <= 0 {
		return
	}
	remaining := len(sl.prompt) - int(sl.nPast)
	n := min(remaining, space)
	// End the batch at the next checkpoint, so the sequence can be pooled at
	// exactly that position before anything past it is decoded.
	if len(sl.checkpoints) > 0 && sl.checkpoints[0].pos > sl.nPast {
		n = min(n, int(sl.checkpoints[0].pos-sl.nPast))
	}

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
			// Pool it once the decode lands — unless the pool adopts only whole
			// entries, which no later prompt makes of this one: it would have to
			// repeat this reply's opening. That pool is fed at the checkpoint.
			if !s.pool.wholeOnly() {
				sl.publishLen = len(sl.prompt)
			}
		}
	}
	sl.nPast += int32(n)
	if len(sl.checkpoints) > 0 && sl.nPast == sl.checkpoints[0].pos {
		sl.publishLen = int(sl.nPast)
		sl.publishShared = sl.checkpoints[0].shared
		sl.checkpoints = sl.checkpoints[1:]
	}
}

// addDecode queues the token a generating slot reads this step, and the
// drafts that follow it, each with its logits to check the draft after it by.
func (s *scheduler) addDecode(sl *slot) {
	room := s.batch.Cap() - s.batch.Len()
	if room <= 0 {
		sl.drafts = sl.drafts[:0]
		return
	}
	sl.drafts = sl.drafts[:min(len(sl.drafts), room-1)]
	idx := s.batch.Add(sl.next, sl.nPast, sl.seq, true)
	for i, d := range sl.drafts {
		s.batch.Add(d, sl.nPast+1+int32(i), sl.seq, true)
	}
	sl.logitIdx = idx
	sl.nPast += 1 + int32(len(sl.drafts))
}

// allowance is how many tokens may be drafted for a generating slot: as many
// as still fit its context and its reply's budget after the one it reads.
func (s *scheduler) allowance(sl *slot) int {
	if sl.media > 0 {
		return 0 // the head did not read the pictures
	}
	return min(s.ctxPerSeq-int(sl.nPast)-1, sl.maxGen-sl.nGen-1)
}

// harvest samples a slot's next token and decides whether it is finished —
// and with drafts, the token after each the model agrees with, until one it
// does not: the token it wrote there instead is the last this step.
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

	drafts := sl.drafts
	sl.drafts = sl.drafts[:0]
	first := sl.nPast - 1 - int32(len(drafts)) // the position read at idx
	n, goesOn := checkDrafts(drafts,
		func(i int) llama.Token {
			var t0 time.Time
			if s.debug {
				t0 = time.Now()
			}
			var tok llama.Token
			if i == 0 && sl.hasPre {
				tok, sl.hasPre = sl.pre, false
			} else {
				tok = sl.sampler.Sample(s.lctx, idx+int32(i))
			}
			if s.debug {
				s.tSample += time.Since(t0)
			}
			return tok
		},
		func(tok llama.Token, i int) bool { return s.take(sl, tok, first+int32(i)+1) })
	s.drafted.Add(int64(len(drafts)))
	s.draftsAccepted.Add(int64(n))
	if !goesOn || n == len(drafts) {
		return
	}
	// The model wrote something else after the n-th draft, so what it read past
	// that was the wrong text: back to just after it.
	last := first + int32(n)
	if s.draft != nil {
		s.draft.settle(sl.seq, last, idx+int32(n))
	}
	if !llama.TrimSequence(s.lctx, sl.seq, last+1) {
		s.release(sl, fmt.Errorf("the model could not go back over %d rejected drafts", len(drafts)-n))
		return
	}
	sl.nPast = last + 1
}

// take adds a token to a slot's reply and reports whether the slot goes on,
// false when the token ended it. past is the slot's positions in the cache
// with the token it was sampled at.
func (s *scheduler) take(sl *slot, tok llama.Token, past int32) bool {
	if llama.IsEOG(s.vocab, tok) {
		s.finishSlot(sl)
		return false
	}

	piece := llama.TokenToPiece(s.vocab, tok, false)
	if piece == "" {
		s.finishSlot(sl)
		return false
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
		return false
	}
	if sl.nGen >= sl.maxGen || int(past) >= s.ctxPerSeq {
		sl.job.stats.Truncated = true
		s.finishSlot(sl)
		return false
	}
	return true
}

// finishSlot ends a reply normally, flushing whatever was held back.
func (s *scheduler) finishSlot(sl *slot) {
	text, _ := s.tpl.TrimStop(sl.out.String())
	if len(text) > sl.emitted {
		// Never blocking here keeps a client that stopped reading from
		// stalling every other slot. A reply longer than the buffer, read
		// slowly, fills it; what does not fit then goes as the job's tail,
		// which the reader takes after the rest, instead of being lost.
		select {
		case sl.job.frags <- text[sl.emitted:]:
		default:
			sl.job.tail = text[sl.emitted:]
		}
		sl.emitted = len(text)
	}
	s.release(sl, nil)
}

// release frees a slot and tells its caller the reply is over.
func (s *scheduler) release(sl *slot, err error) {
	if sl.job == nil {
		return
	}
	if s.lctx != 0 {
		s.forget(sl.seq)
	}
	if sl.sampler != nil {
		if held, n := sl.sampler.Held(); held {
			s.callsHeld.Add(1)
			s.redrawn.Add(int64(n))
		}
		sl.sampler.Close()
		sl.sampler = nil
	}

	// Every ending comes through here — the model stopping, the budget running
	// out, a cancelled request, a dead backend — so this is the one place the
	// measurements have to be handed over. A request that died during prefill
	// reports its prompt and a zero eval, which is what happened.
	sl.job.stats.PromptTokens = len(sl.prompt) + sl.media
	sl.job.stats.EvalTokens = sl.nGen
	sl.job.stats.PromptEvalDuration = sl.promptEval
	if !sl.evalStart.IsZero() {
		sl.job.stats.EvalDuration = time.Since(sl.evalStart)
	}

	sl.job.finish(err)
	s.rate.done()
	s.busy.Add(-1)
	s.promptSeen.Add(int64(len(sl.prompt) + sl.media))
	s.reusedSeen.Add(int64(sl.reused))
	s.evalSeen.Add(int64(sl.nGen))
	sl.job = nil
	sl.prompt = nil
	sl.drafts = sl.drafts[:0]
	sl.logitIdx = -1
	sl.admitted = time.Time{}
	sl.evalStart = time.Time{}
	sl.out.Reset()
}

// compact keeps the slots in use numbered from 0 on, with no gap, when each
// sequence has a stream of the cache to itself — with no prefix pool, which is
// what asks for one buffer shared by all (see newContext). llama.cpp reads the
// sequences of such a cache through the model together only while their ids
// run on unbroken, and splits a batch at every gap: at 128 replies of
// Qwen2.5-0.5B on the M3 Ultra a step took 21 ms while every slot was busy,
// then 36 and 60 as replies ended and left gaps. So the highest slot in use
// moves into the lowest free one: llama.cpp copies its stream (whole, as it
// copies across streams — for a 0.5B model's 1024-token slot, 12 MB), and the
// two slots trade places and sequence ids. admit has already given the free
// slots below to requests waiting for one, which is cheaper still.
func (s *scheduler) compact() {
	if s.pool != nil || s.lctx == 0 {
		return
	}
	for {
		free, last := -1, -1
		for i, sl := range s.slots {
			switch {
			case sl.job != nil:
				last = i
			case free < 0:
				free = i
			}
		}
		if free < 0 || last < free {
			return
		}
		from, to := s.slots[last], s.slots[free]
		llama.CopySequence(s.lctx, from.seq, to.seq, -1, -1)
		llama.ForgetSequence(s.lctx, from.seq, -1, -1)
		s.draft.move(from.seq, to.seq)
		from.seq, to.seq = to.seq, from.seq
		s.slots[free], s.slots[last] = from, to
		s.moved++
	}
}

// forget drops a sequence's cells, from the head's cache too.
func (s *scheduler) forget(seq int32) {
	llama.ForgetSequence(s.lctx, seq, -1, -1)
	s.draft.forget(seq)
}

// ErrNotStarted means a request was still queued when the engine went down, so
// nothing of it ever reached llama.cpp or the client.
//
// It exists to separate the two halves of a shutdown. A slot that was mid-batch
// has already streamed part of an answer, and re-running it would repeat that
// text; it can only go on from where it stopped (ErrInterrupted). A job still
// in the queue holds only its messages and parameters — no KV state, no emitted
// tokens, no tie to the context that died — and can simply be run again on the
// replacement.
//
// That distinction is most of the blast radius. When a backend failure retired
// a model mid-load-test, 26 requests failed at once and the majority of them
// had never started. Callers that can reload a model should retry this; callers
// that cannot should report it. Engine cannot reload itself, so it reports.
var ErrNotStarted = errors.New("the engine went down before this request started")

// ErrInterrupted means the backend failed while a request was running, ending
// the context. What the reply had come to reached the caller with the error, so
// a caller able to reload the model can have the replacement go on from there
// (GenParams.Resume) instead of failing it: nothing is sent twice.
//
// On a Mac the usual cause is the GPU running out of memory under the model,
// because something else on the machine started using it after the load.
var ErrInterrupted = errors.New("the engine went down in the middle of this reply")

// drain ends every in-flight and queued request on shutdown.
//
// The two loops fail their jobs with different errors on purpose — see
// ErrNotStarted.
func (s *scheduler) drain() {
	closing := errors.New("engine is closing")
	for _, sl := range s.slots {
		s.release(sl, closing)
	}
	if s.held != nil {
		s.held.finish(ErrNotStarted)
		s.hold(nil)
	}
	s.drainQueue()
}

// drainQueue answers every request still queued: nothing of it happened.
func (s *scheduler) drainQueue() {
	for {
		select {
		case j := <-s.incoming:
			s.taken(j)
			j.finish(ErrNotStarted)
		default:
			return
		}
	}
}

// ended is run's end, however it came: from now on submit refuses, and what
// was queued after the loop's last drain, before it could know, is answered.
// What is queued later still is answered by submit itself.
func (s *scheduler) ended() {
	close(s.stopped)
	s.drainQueue()
}
