package mlxengine

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

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

// Inkling is one loaded Inkling 2 model: Thinking Machines' looped
// transformer (mlxmodel.Inkling2). Replies write together, a token each a
// step (mlxmodel.Inkling2Batch): a step of this model costs little more for
// four rows than for one.
//
// It is an engine of its own beside Engine, which is written around the
// Qwen3.5 family's cache and batch: the two share the sampling, the pool of
// read prompts and the way a step is recorded ahead of the one in flight,
// and nothing of Engine changes for it.
type Inkling struct {
	path  string
	model *mlxmodel.Inkling2
	tok   *tokenizer.Tokenizer
	marks inkMarks
	// pictureTok is the token a picture's places hold, when the model has
	// its vision tower.
	pictureTok int32
	slots      int
	ctx        int
	opts       engine.Options

	jobs chan *inkJob
	quit chan struct{}
	done chan struct{}
	once sync.Once
	pool *prefixPool

	waiting, busy      atomic.Int32
	promptTok, evalTok atomic.Int64
	reusedTok          atomic.Int64
	broken             atomic.Bool
}

type inkJob struct {
	ctx    context.Context
	prompt []int32
	// pictures are the prompt's pictures: where a picture's tokens are, the
	// prompt holds the picture's own number (inkPictureID).
	pictures []*inkPicture
	// marks are where the prompt is pooled as it is read: the end of what
	// every conversation of the agent opens with, and its own end.
	marks  []checkpoint
	params engine.GenParams
	frags  chan string
	done   chan struct{}
	stats  engine.Stats
	err    error

	reply    inkReply
	out      []byte
	emitted  int
	gen      []int32
	maxGen   int
	admitted time.Time
	firstTok time.Time
	deadline time.Time
	rng      *rand.Rand
	last     int32
}

// IsInkling reports whether path is a directory holding an Inkling 2 model.
func IsInkling(path string) bool {
	return IsModelDir(path) && mlxmodel.ServesInkling2(path)
}

// InklingReadsPictures reports whether path holds an Inkling 2 model with its
// vision tower.
func InklingReadsPictures(path string) bool {
	if !IsInkling(path) {
		return false
	}
	_, err := os.Stat(filepath.Join(path, "model.visual.proj_out.weight.safetensors"))
	return err == nil
}

// OpenInkling loads the Inkling 2 model in dir.
func OpenInkling(dir string, opts engine.Options) (*Inkling, error) {
	// The MLX libraries are fetched the way Engine fetches them; llama.cpp's
	// is bound too, as it is for every engine, though no template is
	// rendered through it here.
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
	e := &Inkling{path: dir, tok: tok, opts: opts, ctx: opts.ContextSize, slots: opts.Slots,
		jobs: make(chan *inkJob, 1024), quit: make(chan struct{}), done: make(chan struct{})}
	for name, id := range map[string]*int32{
		inkModel: &e.marks.model, inkText: &e.marks.text, inkThinking: &e.marks.thinking, inkInvoke: &e.marks.invoke,
		inkToolText: &e.marks.toolText, inkToolError: &e.marks.toolError, inkEnd: &e.marks.end, inkEndSampling: &e.marks.endSampling,
	} {
		v, ok := tok.ID(name)
		if !ok {
			return nil, fmt.Errorf("%s: its tokenizer has no %s", dir, name)
		}
		*id = v
	}
	start := time.Now()
	if e.model, err = mlxmodel.LoadInkling2(dir); err != nil {
		return nil, err
	}
	if e.model.ReadsPictures() {
		name := "<|unused_200054|>"
		var pc struct {
			ImageToken string `json:"image_token"`
		}
		if b, err := os.ReadFile(filepath.Join(dir, "processor_config.json")); err == nil && json.Unmarshal(b, &pc) == nil && pc.ImageToken != "" {
			name = pc.ImageToken
		}
		id, ok := tok.ID(name)
		if !ok {
			e.model.Free()
			return nil, fmt.Errorf("%s: its tokenizer has no %s for a picture's place", dir, name)
		}
		e.pictureTok = id
	}
	mlx.SetCacheLimit(cacheLimit)
	if e.ctx <= 0 {
		e.ctx = defaultCtx
	}
	if e.slots <= 0 {
		e.slots = defaultSlots
	}
	prefix := opts.PrefixSlots
	if prefix == 0 {
		prefix = engine.DefaultPrefixSlots
	}
	e.pool = newPrefixPool(prefix)
	log.Printf("mlx: %s loaded in %v — Inkling 2, %d layers applied %d times a token, %.1f GB, %d slots × %d tokens, %d prefix entries",
		filepath.Base(dir), time.Since(start).Round(time.Millisecond), e.model.Config.NumLayers, e.model.Visits(),
		float64(e.model.Bytes())/1e9, e.slots, e.ctx, max(prefix, 0))
	if e.model.ReadsPictures() {
		log.Printf("mlx: %s: reads pictures, a token for every %d × %d pixels, at most %d a picture", filepath.Base(dir), e.model.PictureTile(), e.model.PictureTile(), inkPictureTokens)
	}
	if e.model.DenseBits > 0 {
		log.Printf("mlx: %s: the attention's projections and the output layer are packed to %d bits, which is quicker to read; KINFER_INK_DENSE=off keeps them as published",
			filepath.Base(dir), e.model.DenseBits)
	}
	go e.run()
	return e, nil
}

// Chat is ChatFull without the statistics.
func (e *Inkling) Chat(ctx context.Context, msgs []chat.Message, p engine.GenParams, onToken func(string)) (string, error) {
	s, _, err := e.ChatFull(ctx, msgs, p, onToken)
	return s, err
}

// encode is a prompt's tokens: the template's own stretches read with their
// control tokens, what a caller wrote as text whatever it spells, and for a
// picture as many places as it has tiles.
func (e *Inkling) encode(p inkPrompt) (ids []int32, pictures []*inkPicture, err error) {
	for _, x := range p {
		if x.picture == nil {
			ids = append(ids, e.tok.Encode(x.text, !x.given)...)
			continue
		}
		if !e.model.ReadsPictures() {
			return nil, nil, engine.PictureError{Err: fmt.Errorf("this Inkling 2 reads no pictures: it has no vision tower")}
		}
		pic := &inkPicture{at: len(ids)}
		if pic.pixels, pic.rows, pic.cols, err = inkPixels(x.picture, e.model.PictureTile()); err != nil {
			return nil, nil, err
		}
		pic.n = pic.rows * pic.cols
		id := inkPictureID(x.picture)
		for i := 0; i < pic.n; i++ {
			ids = append(ids, id)
		}
		pictures = append(pictures, pic)
	}
	return ids, pictures, nil
}

func thinkingOf(p engine.GenParams) chat.Thinking {
	return chat.Thinking{Off: p.NoThink, On: p.Think, Effort: p.Effort, KeepAll: p.KeepThinking}
}

// ChatFull generates a reply to msgs, streaming it to onToken.
func (e *Inkling) ChatFull(ctx context.Context, msgs []chat.Message, p engine.GenParams, onToken func(string)) (string, engine.Stats, error) {
	th := thinkingOf(p)
	rendered, err := inkRender(msgs, p.Tools, th)
	if err != nil {
		return "", engine.Stats{}, err
	}
	prompt, pictures, err := e.encode(rendered)
	if err != nil {
		return "", engine.Stats{}, err
	}
	j := &inkJob{ctx: ctx, prompt: prompt, pictures: pictures, params: p, frags: make(chan string, 1<<16), done: make(chan struct{})}
	// What the conversations of one agent share: its tools, its system
	// messages and the effort line after them.
	n := 0
	for n < len(msgs) && msgs[n].Role == "system" {
		n++
	}
	if opening, err := inkRender(msgs[:n], p.Tools, th); err == nil {
		if ids, _, err := e.encode(opening); err == nil && len(ids) < len(j.prompt) {
			j.marks = append(j.marks, checkpoint{pos: len(ids), shared: true})
		}
	}
	j.marks = append(j.marks, checkpoint{pos: len(j.prompt)})

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

// CountTokens is how many tokens msgs is as a prompt.
func (e *Inkling) CountTokens(msgs []chat.Message, p engine.GenParams) (int, error) {
	rendered, err := inkRender(msgs, p.Tools, thinkingOf(p))
	if err != nil {
		return 0, err
	}
	ids, _, err := e.encode(rendered)
	return len(ids), err
}

func (e *Inkling) Broken() bool                       { return e.broken.Load() }
func (e *Inkling) ToolFormat() tools.Format           { return tools.Hermes }
func (e *Inkling) PromptTokens() int64                { return e.promptTok.Load() }
func (e *Inkling) EvalTokens() int64                  { return e.evalTok.Load() }
func (e *Inkling) Drafts() (drafted, accepted int64)  { return 0, 0 }
func (e *Inkling) ToolGrammar() (held, redrawn int64) { return 0, 0 }
func (e *Inkling) Context() (perSlot, most int)       { return e.ctx, e.ctx }
func (e *Inkling) Path() string                       { return e.path }

// Reuse is the prompt tokens the pool of read prompts supplied, and how many
// prompts it may hold.
func (e *Inkling) Reuse() (tokens int64, entries int) {
	if e.pool == nil {
		return e.reusedTok.Load(), 0
	}
	return e.reusedTok.Load(), len(e.pool.entries)
}

// Load is the requests waiting, the ones being answered, and how many can be
// at once.
func (e *Inkling) Load() (waiting, busy, slots int) {
	return int(e.waiting.Load()), int(e.busy.Load()), e.slots
}

// Close ends the engine and frees the model.
func (e *Inkling) Close() {
	e.once.Do(func() { close(e.quit) })
	<-e.done
}

// run is the one goroutine that uses MLX.
func (e *Inkling) run() {
	defer close(e.done)
	batch := e.model.NewBatch()
	// rows[r] is the reply in batch row r, nil for a free row.
	var rows []*inkJob
	// pending is the step in flight: its tokens, still on the GPU, one a
	// row, are the next step's input before the host has seen them.
	var pending mlx.Array
	closed := fmt.Errorf("engine for %s closed", e.path)
	end := func(err error) {
		for _, j := range rows {
			if j != nil {
				e.finish(j, err)
			}
		}
		rows = nil
		if pending != 0 {
			pending.Eval()
			pending.Free()
			pending = 0
		}
		batch.Free()
	}
	defer func() {
		end(closed)
		for {
			select {
			case j := <-e.jobs:
				e.waiting.Add(-1)
				e.finish(j, closed)
				continue
			default:
			}
			break
		}
		e.pool.free()
		e.model.Free()
	}()
	fail := func(err error) {
		e.broken.Store(e.broken.Load() || mlxFault(err))
		end(err)
	}
	// settle reads the step in flight and hands each row its token.
	settle := func() {
		if pending == 0 {
			return
		}
		toks, err := readTokens(pending)
		pending.Free()
		pending = 0
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
			var j *inkJob
			if active() == 0 {
				select {
				case j = <-e.jobs:
				case <-e.quit:
					return
				}
			} else {
				select {
				case j = <-e.jobs:
				case <-e.quit:
					return
				default:
				}
			}
			if j == nil {
				break
			}
			e.waiting.Add(-1)
			// The newcomer's prompt is read alone and its first token is on
			// the host, so the step in flight is finished first.
			settle()
			c := e.admit(j)
			if c == nil {
				continue
			}
			r, err := batch.Add(c, min(e.slots, active()+1+len(e.jobs)))
			if err != nil {
				c.Free()
				e.broken.Store(e.broken.Load() || mlxFault(err))
				e.finish(j, err)
				continue
			}
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
			moved, err := batch.Compact()
			if err != nil {
				fail(err)
				continue
			}
			kept := make([]*inkJob, len(moved))
			for old, nu := range moved {
				kept[nu] = rows[old]
			}
			rows = kept
			if active() == 0 {
				continue
			}
		}

		next, err := e.step(batch, rows, pending)
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

// leave gives each row its token and releases the rows whose reply ended.
func (e *Inkling) leave(batch *mlxmodel.Inkling2Batch, rows []*inkJob, toks []int32) {
	for r, j := range rows {
		if j != nil && !e.accept(j, toks[r]) {
			batch.Release(r)
			rows[r] = nil
		}
	}
}

// admit readies j and reads its prompt; it returns the cache for the batch,
// j's first token taken, or nil when the reply is already over.
func (e *Inkling) admit(j *inkJob) *mlxmodel.Inkling2Cache {
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
	j.reply = inkReply{marks: e.marks, bytes: func(t int32) []byte { return e.tok.Bytes([]int32{t}) }}
	j.stats.PromptTokens = len(j.prompt)
	j.maxGen = e.ctx - len(j.prompt)
	if j.params.MaxTokens > 0 {
		j.maxGen = min(j.maxGen, j.params.MaxTokens)
	}
	if j.maxGen <= 0 {
		j.stats.Truncated = true
		e.finish(j, nil)
		return nil
	}
	c, first, err := e.prefill(j)
	if err != nil {
		e.broken.Store(e.broken.Load() || mlxFault(err))
		e.finish(j, err)
		return nil
	}
	j.stats.PromptEvalDuration = time.Since(j.admitted)
	j.firstTok = time.Now()
	if !e.accept(j, first) {
		c.Free()
		return nil
	}
	return c
}

// mlxFault says whether err is MLX's own, after which the model's state is
// not to be trusted.
func mlxFault(err error) bool {
	_, ok := err.(*mlx.Error)
	return ok
}

func readTokens(a mlx.Array) (toks []int32, err error) {
	defer mlx.Recover(&err)
	return a.Int32s(), nil
}

// prefill reads j's prompt into a cache and samples the reply's first token.
func (e *Inkling) prefill(j *inkJob) (c *mlxmodel.Inkling2Cache, first int32, err error) {
	defer mlx.Recover(&err)
	c, read := e.model.NewCache(), 0
	if p := e.pool.match(j.prompt); p != nil {
		c.Free()
		c, read = p.cache.(*mlxmodel.Inkling2Cache).Clone(), len(p.tokens)
		e.reusedTok.Add(int64(read))
		j.stats.CachedTokens = read
	}
	defer func() {
		if err != nil {
			c.Free()
			c = nil
		}
		for _, pic := range j.pictures {
			if pic.features != 0 {
				pic.features.Free()
				pic.features = 0
			}
			pic.pixels = nil
		}
	}()
	marks := j.marks
	for len(marks) > 0 && marks[0].pos <= read {
		marks = marks[1:]
	}
	var logits mlx.Array
	for read < len(j.prompt) {
		end := min(read+inkPrefillChunk, len(j.prompt))
		if len(marks) > 0 && marks[0].pos < end {
			end = marks[0].pos
		}
		if logits != 0 {
			logits.Free()
		}
		if len(j.pictures) == 0 {
			logits, err = e.model.Forward([][]int32{j.prompt[read:end]}, c)
		} else {
			toks, spans, free, perr := e.pictureSpans(j, read, end)
			if err = perr; err == nil {
				logits, err = e.model.ForwardSpans(toks, spans, c)
			}
			free()
		}
		if err != nil {
			return c, 0, err
		}
		read = end
		if len(marks) > 0 && read == marks[0].pos {
			e.pool.put(j.prompt[:read], marks[0].shared, func() held { return c.Clone() })
			marks = marks[1:]
		}
	}
	e.promptTok.Add(int64(len(j.prompt)))
	if logits == 0 {
		// The whole prompt was pooled: its last token is read again for the
		// scores after it, from the entry before it if there is one.
		return c, 0, fmt.Errorf("a prompt read in full by another request has no scores of its own")
	}
	s := mlx.NewScope()
	defer s.Close()
	tok := sampleGraph(logits, []sampling.Params{j.params.Params}, [][]int32{nil}, []*rand.Rand{j.rng})
	first = tok.Int32s()[0]
	logits.Free()
	return c, first, nil
}

// inkStepBuffers has a step encoded into step-sized command buffers
// (mlx.AsyncEvalStep), as Engine's are.
var inkStepBuffers = os.Getenv("KINFER_INK_STEP_BUFFERS") != "off"

// inkPrefillChunk is how many prompt tokens are read in one pass. A pass
// scores every one of its tokens against the keys in view, 24 heads of each
// layer application, in float32: 512 keeps that under a hundred megabytes,
// and read 1,500 tokens a fifth sooner than 256 did.
const inkPrefillChunk = 512

// inkStepRows is the most rows a step is encoded for into step-sized command
// buffers (mlx.AsyncEvalStep); a wider one keeps the GPU busy as it is.
const inkStepRows = 16

// step records one decode step for rows and starts it, returning its tokens
// unevaluated, one a row. in is the step before it when that is still on the
// GPU, or 0 to take each row's last token from the host.
func (e *Inkling) step(b *mlxmodel.Inkling2Batch, rows []*inkJob, in mlx.Array) (next mlx.Array, err error) {
	defer mlx.Recover(&err)
	s := mlx.NewScope()
	defer s.Close()
	// A free row is computed along with the rest and its token ignored; it
	// borrows a live row's sampling so the batch samples as one group.
	var model *inkJob
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
		params[r], hist[r], rngs[r] = j.params.Params, j.history(), j.rng
	}
	next = sampleGraph(logits, params, hist, rngs)
	s.Keep(next)
	if out := append([]mlx.Array{next}, b.Arrays()...); inkStepBuffers && len(rows) <= inkStepRows {
		mlx.AsyncEvalStep(out...)
	} else {
		mlx.AsyncEval(out...)
	}
	return next, nil
}

// history is what the penalties read: the reply so far.
func (j *inkJob) history() []int32 {
	if n := j.params.RepeatLastN; n > 0 && len(j.gen) > n {
		return j.gen[len(j.gen)-n:]
	}
	return j.gen
}

// accept takes the reply's next token, sends whatever text it completes, and
// reports whether the reply goes on.
func (e *Inkling) accept(j *inkJob, tok int32) bool {
	if err := j.ctx.Err(); err != nil {
		e.finish(j, err)
		return false
	}
	text, done := j.reply.take(tok)
	if !done {
		j.gen = append(j.gen, tok)
		j.last = tok
		e.evalTok.Add(1)
	}
	j.out = append(j.out, text...)
	if pending := string(j.out[j.emitted:]); pending != "" {
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
	case done:
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

// finish ends j: the rest of its text, its statistics, its error.
func (e *Inkling) finish(j *inkJob, err error) {
	if !j.admitted.IsZero() {
		e.busy.Add(-1)
	}
	if rest := strings.ToValidUTF8(string(j.out[j.emitted:]), "�"); rest != "" {
		select {
		case j.frags <- rest:
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
