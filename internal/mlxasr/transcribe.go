package mlxasr

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// Result is a transcript and what it cost.
type Result struct {
	Text           string // what the model wrote after the prompt, "language …<asr_text>" included unless forced
	PromptTokens   int
	Tokens         int
	Encode         time.Duration // features and the audio encoder
	Prefill        time.Duration
	Decode         time.Duration
	Truncated      bool // stopped at maxTokens
	AudioTokens    int
	ForcedLanguage string
}

// prefillChunk is how many positions a prefill pass reads, as mlx-lm's
// prefill_step_size.
const prefillChunk = 2048

// prompt is mlx-audio's _build_prompt: the system message (what is known
// about the recording, or empty), the sound, and the assistant's turn opened —
// with "language X<asr_text>" when a language is forced. The system text is
// tokenized as text, so it cannot open a turn of its own.
func (m *Model) prompt(system, language string, audio int) (pre, post []int32) {
	enc := func(s string, special bool) []int32 { return m.tok.Encode(s, special) }
	pre = append(pre, enc("<|im_start|>system\n", true)...)
	if system != "" {
		pre = append(pre, enc(system+"\n", false)...)
	}
	pre = append(pre, enc("<|im_end|>\n<|im_start|>user\n<|audio_start|>", true)...)
	post = append(post, enc("<|audio_end|><|im_end|>\n<|im_start|>assistant\n", true)...)
	if language != "" {
		post = append(post, enc("language "+language+"<asr_text>", true)...)
	}
	return pre, post
}

// Language is the model's spelling of a language, or "" for one it does not
// list.
func (m *Model) Language(name string) string { return m.languages[strings.ToLower(name)] }

// Transcribe writes down what pcm, 16 kHz mono, says. language, when not "",
// starts the reply as Qwen3-ASR is told one; onToken, when not nil, gets the
// text as it is written.
func (m *Model) Transcribe(ctx context.Context, pcm []float32, system, language string, maxTokens int, onToken func(string)) (res Result, err error) {
	defer mlx.Recover(&err)
	if language != "" {
		if l := m.Language(language); l != "" {
			language = l
		}
	}
	res.ForcedLanguage = language
	debug := os.Getenv("KINFER_ASR_DEBUG") != ""
	mark := func(what string) {
		if debug {
			u := mlx.MemoryUse()
			log.Printf("mlxasr: %-8s active %.2f GB peak %.2f GB cache %.2f GB", what, float64(u.Active)/1e9, float64(u.Peak)/1e9, float64(u.Cache)/1e9)
			mlx.ResetPeak()
		}
	}
	mark("start")
	if debug {
		checkpoint = func(what string, a mlx.Array) { mlx.Eval(a); mark(what) }
		defer func() { checkpoint = nil }()
	}
	t0 := time.Now()

	// The sound, encoded.
	var audio mlx.Array
	func() {
		s := mlx.NewScope()
		defer s.Close()
		feats, T, ferr := logMel(pcm, m.cfg.Thinker.Audio.Mels)
		if ferr != nil {
			err = ferr
			return
		}
		if checkpoint != nil {
			checkpoint("features", feats)
		}
		audio = m.enc.forward(feats, T)
		s.Escape(audio)
		mlx.Eval(audio)
	}()
	if err != nil {
		return res, err
	}
	defer audio.Free()
	n := audio.Shape()[0]
	res.AudioTokens = n
	res.Encode = time.Since(t0)
	mark("encoded")

	// The prompt, the sound where its <|audio_pad|> tokens go.
	pre, post := m.prompt(system, language, n)
	res.PromptTokens = len(pre) + n + len(post)
	cache := m.dec.newCache()
	defer cache.free()
	t1 := time.Now()
	var logits mlx.Array
	func() {
		s := mlx.NewScope()
		defer s.Close()
		a := m.dec.embed.rows(mlx.FromInt32(pre, 1, len(pre)))
		b := m.dec.embed.rows(mlx.FromInt32(post, 1, len(post)))
		h := mlx.Concatenate(1, a, mlx.Astype(mlx.Reshape(audio, 1, n, m.cfg.Thinker.Text.Hidden), a.Dtype()), b)
		s.Escape(h)
		L := res.PromptTokens
		for start := 0; start < L; start += prefillChunk {
			end := min(start+prefillChunk, L)
			part := mlx.Slice(h, []int{0, start, 0}, []int{1, end, m.cfg.Thinker.Text.Hidden})
			logits = m.dec.step(part, cache) // the scope frees the passes' logits but the last
			mlx.Eval(append([]mlx.Array{logits}, cache.arrays()...)...)
		}
		s.Escape(logits)
		h.Free()
	}()
	res.Prefill = time.Since(t1)
	mark("prefill")

	// Greedily, one token a step, the next step recorded before the last is
	// read so the GPU does not wait for the host.
	t2 := time.Now()
	next := func(logits mlx.Array) mlx.Array {
		s := mlx.NewScope()
		defer s.Close()
		y := mlx.Astype(mlx.Argmax(logits, -1, false), mlx.Int32)
		s.Keep(y)
		return y
	}
	y := next(logits)
	logits.Free()
	mlx.AsyncEval(y)
	var ids []int32
	var shown []byte
	emit := func() {
		if onToken == nil {
			return
		}
		b := m.tok.Bytes(ids)
		if !utf8.Valid(b) {
			return // a character split across tokens: wait for the rest
		}
		if len(b) > len(shown) {
			onToken(string(b[len(shown):]))
			shown = b
		}
	}
	for {
		var ny mlx.Array
		if len(ids)+1 < maxTokens {
			s := mlx.NewScope()
			lg := m.dec.step(m.dec.embed.rows(mlx.Reshape(y, 1, 1)), cache)
			ny = mlx.Astype(mlx.Argmax(lg, -1, false), mlx.Int32)
			s.Keep(ny)
			s.Close()
			// Only the token is asked for, as mlx-lm asks: the cache is computed
			// as what the token needs. Naming the cache's arrays too cost 8% on a
			// long reply — 185 tok/s against 201 over 405 tokens after a prompt
			// of 2,148.
			mlx.AsyncEval(ny)
		}
		tok := y.Int32s()[0]
		y.Free()
		if m.eos[tok] {
			if ny != 0 {
				ny.Free()
			}
			break
		}
		ids = append(ids, tok)
		emit()
		if ny == 0 {
			res.Truncated = true
			break
		}
		if ctx.Err() != nil {
			ny.Free()
			return res, ctx.Err()
		}
		y = ny
	}
	res.Tokens = len(ids)
	res.Decode = time.Since(t2)
	mark("decoded")
	res.Text = m.tok.Decode(ids)
	// Let go of the buffers MLX keeps for reuse, as mlx-audio does after each
	// recording: the next one is of other shapes, and kept they are memory the
	// process holds for nothing.
	mlx.ClearCache()
	if res.Text == "" && len(ids) > 0 {
		return res, fmt.Errorf("the transcript decoded to nothing")
	}
	return res, nil
}
