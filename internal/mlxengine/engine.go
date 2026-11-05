// Package mlxengine serves a model from a directory of safetensors on MLX —
// the same requests, replies and statistics as internal/engine gives a GGUF
// on llama.cpp, so the server need not know which one it holds.
//
// One goroutine owns MLX. Requests reach it over a channel; each reads its
// prompt alone, then joins the batch that decodes one token for every reply
// per step, and leaves it when its reply ends.
package mlxengine

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/mlxmodel"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
	"github.com/LocalKinAI/kinfer/internal/sampling"
	"github.com/LocalKinAI/kinfer/internal/tokenizer"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// Defaults when Options leave them to the engine.
const (
	defaultSlots = 8
	defaultCtx   = 32768
	prefillChunk = 2048
	cacheLimit   = 4 << 30
)

// IsModelDir reports whether path is a directory MLX can serve: a
// config.json, a tokenizer.json and safetensors.
func IsModelDir(path string) bool {
	st, err := os.Stat(path)
	if err != nil || !st.IsDir() {
		return false
	}
	for _, f := range []string{"config.json", "tokenizer.json"} {
		if _, err := os.Stat(filepath.Join(path, f)); err != nil {
			return false
		}
	}
	m, _ := filepath.Glob(filepath.Join(path, "*.safetensors"))
	return len(m) > 0
}

// Engine is one loaded MLX model.
type Engine struct {
	path  string
	model *mlxmodel.Qwen35
	tok   *tokenizer.Tokenizer
	tpl   *chat.Template
	eos   map[int32]bool
	slots int
	ctx   int
	opts  engine.Options

	jobs chan *job
	quit chan struct{}
	done chan struct{}
	once sync.Once

	// prof, when KINFER_MLX_PROFILE is set, adds up where the run loop's
	// time goes and logs it every ten seconds.
	prof *profile

	waiting, busy      atomic.Int32
	promptTok, evalTok atomic.Int64
	broken             atomic.Bool
}

type job struct {
	ctx    context.Context
	prompt []int32
	params engine.GenParams
	frags  chan string
	done   chan struct{}
	stats  engine.Stats
	err    error

	out      []byte // the reply so far, as bytes
	emitted  int    // how much of out has been sent
	gen      []int32
	maxGen   int
	admitted time.Time
	firstTok time.Time
	deadline time.Time
	rng      *rand.Rand
	last     int32
}

// Open loads the model in dir.
func Open(dir string, opts engine.Options) (*Engine, error) {
	// The chat template renders through llama.cpp's template engine, so its
	// library is bound even though no GGUF is loaded.
	libDir, err := nativelib.Prepare()
	if err != nil {
		return nil, err
	}
	if err := llama.Bind(libDir); err != nil {
		return nil, err
	}
	libs, err := ensureLibs()
	if err != nil {
		return nil, err
	}
	if err := mlx.Open(libs); err != nil {
		return nil, fmt.Errorf("MLX libraries (%s): %w", libs, err)
	}
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	start := time.Now()
	model, err := mlxmodel.LoadQwen35(dir)
	if err != nil {
		return nil, err
	}

	// MLX keeps freed buffers for reuse up to its whole memory limit. A
	// server's batch and context sizes keep changing, so most of those
	// buffers are of shapes it will not meet again: unchecked they grew past
	// 75 GB beside a 1.7 GB model. A few gigabytes cover the reuse that
	// matters, a decode step's intermediates.
	mlx.SetCacheLimit(cacheLimit)

	tmpl, _ := os.ReadFile(filepath.Join(dir, "chat_template.jinja"))
	e := &Engine{
		path: dir, model: model, tok: tok, opts: opts,
		tpl:   chat.FromTemplate(string(tmpl), dir),
		eos:   map[int32]bool{},
		slots: opts.Slots, ctx: opts.ContextSize,
		jobs: make(chan *job, 1024), quit: make(chan struct{}), done: make(chan struct{}),
	}
	for _, id := range model.Config.EOS() {
		e.eos[id] = true
	}
	for _, s := range []string{"<|im_end|>", "<|endoftext|>"} {
		if id, ok := tok.ID(s); ok {
			e.eos[id] = true
		}
	}
	if e.slots <= 0 {
		e.slots = defaultSlots
	}
	if e.ctx <= 0 {
		e.ctx = defaultCtx
	}
	e.prof = &profile{on: os.Getenv("KINFER_MLX_PROFILE") != "", since: time.Now()}
	log.Printf("mlx: %s loaded in %v — %d layers, %d slots × %d tokens, template %s",
		filepath.Base(dir), time.Since(start).Round(time.Millisecond), model.Config.NumHiddenLayers,
		e.slots, e.ctx, e.tpl.Name)
	go e.run()
	return e, nil
}

// Chat is ChatFull without the statistics.
func (e *Engine) Chat(ctx context.Context, msgs []chat.Message, p engine.GenParams, onToken func(string)) (string, error) {
	s, _, err := e.ChatFull(ctx, msgs, p, onToken)
	return s, err
}

// ChatFull generates a reply to msgs, streaming it to onToken.
func (e *Engine) ChatFull(ctx context.Context, msgs []chat.Message, p engine.GenParams, onToken func(string)) (string, engine.Stats, error) {
	for _, m := range msgs {
		if len(m.Images) > 0 {
			return "", engine.Stats{}, errors.New("pictures are not supported by the MLX backend yet")
		}
	}
	th := chat.Thinking{Off: p.NoThink, Effort: p.Effort}
	prompt := e.tok.Encode(e.tpl.RenderWith(msgs, p.Tools, th), true)
	j := &job{ctx: ctx, prompt: prompt, params: p, frags: make(chan string, 1<<16), done: make(chan struct{})}
	if p.Resume != "" {
		text := p.Resume
		if e.tpl.OpensThink(th) {
			text = strings.TrimPrefix(text, "<think>")
		}
		j.prompt = append(j.prompt, e.tok.Encode(text, false)...)
		j.out, j.emitted = []byte(p.Resume), len(p.Resume)
	} else if e.tpl.OpensThink(th) {
		j.out = []byte("<think>")
	}

	e.waiting.Add(1)
	select {
	case e.jobs <- j:
	case <-e.quit:
		e.waiting.Add(-1)
		return "", engine.Stats{}, fmt.Errorf("engine for %s is closed", e.path)
	}

	var b strings.Builder
	for f := range j.frags {
		b.WriteString(f)
		if onToken != nil {
			onToken(f)
		}
	}
	<-j.done
	return b.String(), j.stats, j.err
}

// CountTokens is how many tokens msgs render to.
func (e *Engine) CountTokens(msgs []chat.Message, p engine.GenParams) (int, error) {
	th := chat.Thinking{Off: p.NoThink, Effort: p.Effort}
	return len(e.tok.Encode(e.tpl.RenderWith(msgs, p.Tools, th), true)), nil
}

func (e *Engine) Broken() bool                      { return e.broken.Load() }
func (e *Engine) ToolFormat() tools.Format          { return e.tpl.ToolFormat() }
func (e *Engine) PromptTokens() int64               { return e.promptTok.Load() }
func (e *Engine) EvalTokens() int64                 { return e.evalTok.Load() }
func (e *Engine) Drafts() (drafted, accepted int64) { return 0, 0 }
func (e *Engine) Context() (perSlot, most int)      { return e.ctx, e.ctx }
func (e *Engine) Path() string                      { return e.path }

// Load is the queue, the replies in progress, and the most there can be.
func (e *Engine) Load() (waiting, busy, slots int) {
	return int(e.waiting.Load()), int(e.busy.Load()), e.slots
}

// Close stops the engine, failing what is queued or running, and frees the
// model.
func (e *Engine) Close() {
	e.once.Do(func() { close(e.quit) })
	<-e.done
}

// run is the one goroutine that uses MLX.
func (e *Engine) run() {
	defer close(e.done)
	batch := e.model.NewBatch()
	// rows[r] is the reply in batch row r, nil for a free row.
	var rows []*job
	defer func() {
		for _, j := range rows {
			if j != nil {
				e.finish(j, fmt.Errorf("engine for %s closed", e.path))
			}
		}
		batch.Free()
		e.model.Free()
	}()

	// pending is the step in flight: its tokens, still on the GPU, one per
	// batch row. While it is pending the next step can already be recorded
	// with those tokens as its input, so the GPU never waits for the host.
	var pending mlx.Array
	defer func() {
		if pending != 0 {
			pending.Free()
		}
	}()
	fail := func(err error) {
		for _, j := range rows {
			if j != nil {
				e.finish(j, err)
			}
		}
		rows = nil
		if pending != 0 {
			pending.Free()
			pending = 0
		}
		batch.Free()
	}
	// settle reads the pending step's tokens and hands each row its own.
	settle := func() {
		if pending == 0 {
			return
		}
		t := time.Now()
		toks, err := readTokens(pending)
		pending.Free()
		pending = 0
		e.prof.add(&e.prof.step, t)
		if err != nil {
			fail(err)
			return
		}
		e.leave(batch, rows, toks)
	}
	active := func() int {
		n := 0
		for _, j := range rows {
			if j != nil {
				n++
			}
		}
		return n
	}

	for {
		// Admit: block for work when idle, otherwise take what has arrived.
		for active() < e.slots {
			var j *job
			if active() == 0 {
				select {
				case j = <-e.jobs:
				case <-e.quit:
					e.drain()
					return
				}
			} else {
				select {
				case j = <-e.jobs:
				case <-e.quit:
					e.drain()
					return
				default:
				}
			}
			if j == nil {
				break
			}
			e.waiting.Add(-1)
			// The newcomer's first token is on the host, so the step in
			// flight is finished first and the next one starts from the host.
			settle()
			t := time.Now()
			c := e.prefill(j)
			e.prof.add(&e.prof.prefill, t)
			if c == nil {
				continue
			}
			t = time.Now()
			// Room for this burst: the rows in use, this one, and the
			// replies queued behind it.
			r, err := batch.Add(c, min(e.slots, active()+1+len(e.jobs)))
			e.prof.add(&e.prof.join, t)
			if err != nil {
				e.finish(j, err)
				continue
			}
			// A batch that grew may have added free rows past r as well.
			for len(rows) < batch.Len() {
				rows = append(rows, nil)
			}
			rows[r] = j
		}
		if active() == 0 {
			settle()
			if active() == 0 && batch.Len() > 0 {
				batch.Free()
				rows = nil
			}
			continue
		}

		// Free rows still cost a step's work each; drop them once they are
		// half the batch and nobody is waiting to take one.
		if free := batch.FreeRows(); free > 0 && free*2 >= batch.Len() && len(e.jobs) == 0 {
			settle()
			t := time.Now()
			moved, err := batch.Compact()
			e.prof.add(&e.prof.leave, t)
			if err != nil {
				fail(err)
				continue
			}
			kept := make([]*job, len(moved))
			for old, nu := range moved {
				kept[nu] = rows[old]
			}
			rows = kept
			if active() == 0 {
				continue
			}
		}

		// Record the next step and start it, then deal with the one before
		// while the GPU runs.
		t := time.Now()
		next, err := e.step(batch, rows, pending)
		e.prof.add(&e.prof.step, t)
		e.prof.tick(active())
		if err != nil {
			fail(err)
			continue
		}
		if pending == 0 {
			pending = next
			continue
		}
		toks, err := readTokens(pending)
		pending.Free()
		pending = next
		if err != nil {
			fail(err)
			continue
		}
		e.leave(batch, rows, toks)
	}
}

func readTokens(a mlx.Array) (toks []int32, err error) {
	defer mlx.Recover(&err)
	return a.Int32s(), nil
}

// leave gives each row its token and releases the rows whose reply ended.
func (e *Engine) leave(batch *mlxmodel.Qwen35Batch, rows []*job, toks []int32) {
	for r, j := range rows {
		if j != nil && !e.accept(j, toks[r]) {
			batch.Release(r)
			rows[r] = nil
		}
	}
}

func (e *Engine) drain() {
	for {
		select {
		case j := <-e.jobs:
			e.waiting.Add(-1)
			e.finish(j, fmt.Errorf("engine for %s closed", e.path))
		default:
			return
		}
	}
}

// prefill reads j's prompt into a fresh cache and takes its first token. It
// returns the cache for the batch, or nil when the reply already ended.
func (e *Engine) prefill(j *job) *mlxmodel.Qwen35Cache {
	if err := j.ctx.Err(); err != nil {
		e.finish(j, err)
		return nil
	}
	e.busy.Add(1)
	j.admitted = time.Now()
	if e.opts.MaxGenerate > 0 {
		j.deadline = j.admitted.Add(e.opts.MaxGenerate)
	}
	seed := j.params.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	j.rng = rand.New(rand.NewSource(seed))
	j.maxGen = e.ctx - len(j.prompt)
	if j.params.MaxTokens > 0 {
		j.maxGen = min(j.maxGen, j.params.MaxTokens)
	}
	j.stats.PromptTokens = len(j.prompt)
	if j.maxGen <= 0 {
		j.stats.Truncated = true
		e.finish(j, nil)
		return nil
	}

	c := e.model.NewCache()
	var logits mlx.Array
	for i := 0; i < len(j.prompt); i += prefillChunk {
		if logits != 0 {
			logits.Free()
		}
		var err error
		logits, err = e.model.Forward([][]int32{j.prompt[i:min(i+prefillChunk, len(j.prompt))]}, c, nil)
		if err != nil {
			c.Free()
			e.finish(j, err)
			return nil
		}
	}
	e.promptTok.Add(int64(len(j.prompt)))
	tok, err := e.sample(logits, []*job{j})
	logits.Free()
	if err != nil {
		c.Free()
		e.finish(j, err)
		return nil
	}
	j.stats.PromptEvalDuration = time.Since(j.admitted)
	j.firstTok = time.Now()
	if !e.accept(j, tok[0]) {
		c.Free()
		return nil
	}
	return c
}

// step records one decode step for rows and starts it, returning its tokens
// unevaluated. in is the previous step's tokens when they are still on the
// GPU, or 0 to take each row's last token from the host.
func (e *Engine) step(b *mlxmodel.Qwen35Batch, rows []*job, in mlx.Array) (next mlx.Array, err error) {
	defer mlx.Recover(&err)
	s := mlx.NewScope()
	defer s.Close()
	// A free row is computed along with the rest and its token ignored; it
	// borrows a live row's sampling so the batch samples as one group.
	var model *job
	for _, j := range rows {
		if j != nil {
			model = j
			break
		}
	}
	var y mlx.Array
	if in != 0 {
		y = mlx.Reshape(in, len(rows), 1)
	} else {
		last := make([]int32, len(rows))
		for r, j := range rows {
			if j != nil {
				last[r] = j.last
			}
		}
		y = mlx.FromInt32(last, len(rows), 1)
	}
	logits := b.Step(y)
	params := make([]sampling.Params, len(rows))
	hist := make([][]int32, len(rows))
	rngs := make([]*rand.Rand, len(rows))
	for r, j := range rows {
		if j == nil {
			j = model
		}
		params[r], hist[r], rngs[r] = j.params.Params, j.gen, j.rng
	}
	next = sampleGraph(logits, params, hist, rngs)
	s.Keep(next)
	mlx.AsyncEval(append([]mlx.Array{next}, b.Arrays()...)...)
	return next, nil
}

func (e *Engine) sample(logits mlx.Array, rows []*job) (toks []int32, err error) {
	defer mlx.Recover(&err)
	s := mlx.NewScope()
	defer s.Close()
	params := make([]sampling.Params, len(rows))
	hist := make([][]int32, len(rows))
	rngs := make([]*rand.Rand, len(rows))
	for r, j := range rows {
		params[r], hist[r], rngs[r] = j.params.Params, j.gen, j.rng
	}
	return sampleRows(logits, params, hist, rngs), nil
}

// accept takes j's next token, sends whatever text it completes, and reports
// whether the reply goes on.
func (e *Engine) accept(j *job, tok int32) bool {
	if err := j.ctx.Err(); err != nil {
		e.finish(j, err)
		return false
	}
	if e.eos[tok] {
		e.finish(j, nil)
		return false
	}
	j.gen = append(j.gen, tok)
	j.last = tok
	e.evalTok.Add(1)
	j.out = append(j.out, e.tok.Bytes([]int32{tok})...)

	text, hitStop := e.tpl.TrimStop(string(j.out))
	if len(text) > j.emitted {
		pending := text[j.emitted:]
		// Hold back a character still arriving: a CJK character spans
		// several tokens, and half of one is not printable.
		if n := validPrefix(pending); n > 0 {
			select {
			case j.frags <- pending[:n]:
				j.emitted += n
			default:
			}
		}
	}
	switch {
	case hitStop:
		e.finish(j, nil)
		return false
	case len(j.gen) >= j.maxGen:
		j.stats.Truncated = true
		e.finish(j, nil)
		return false
	case !j.deadline.IsZero() && time.Now().After(j.deadline):
		j.stats.Timeout = true
		e.finish(j, nil)
		return false
	}
	return true
}

func validPrefix(s string) int {
	n := len(s)
	for n > 0 && !utf8.ValidString(s[:n]) {
		n--
	}
	return n
}

// finish ends j: the rest of its text, its statistics, its error.
func (e *Engine) finish(j *job, err error) {
	if !j.admitted.IsZero() {
		e.busy.Add(-1)
	}
	text, _ := e.tpl.TrimStop(strings.ToValidUTF8(string(j.out), "�"))
	if len(text) > j.emitted {
		select {
		case j.frags <- text[j.emitted:]:
		default:
		}
	}
	j.stats.EvalTokens = len(j.gen)
	if !j.firstTok.IsZero() {
		j.stats.EvalDuration = time.Since(j.firstTok)
	}
	j.err = err
	close(j.frags)
	close(j.done)
}

type profile struct {
	on                         bool
	since                      time.Time
	prefill, join, step, leave time.Duration
	steps, rowSteps            int
}

func (p *profile) add(d *time.Duration, t time.Time) {
	if p.on {
		*d += time.Since(t)
	}
}

func (p *profile) tick(rows int) {
	if !p.on {
		return
	}
	p.steps++
	p.rowSteps += rows
	if el := time.Since(p.since); el > 10*time.Second {
		mem := mlx.MemoryUse()
		log.Printf("mlx profile %v: prefill %v, join %v, decode %v (%d steps, %.1f rows each, %.1f ms a step), leave %v; memory active %.1f GB, cache %.1f GB, peak %.1f GB",
			el.Round(time.Second), p.prefill.Round(time.Millisecond), p.join.Round(time.Millisecond),
			p.step.Round(time.Millisecond), p.steps, float64(p.rowSteps)/float64(max(1, p.steps)),
			float64(p.step.Microseconds())/1000/float64(max(1, p.steps)), p.leave.Round(time.Millisecond),
			float64(mem.Active)/1e9, float64(mem.Cache)/1e9, float64(mem.Peak)/1e9)
		*p = profile{on: true, since: time.Now()}
	}
}
