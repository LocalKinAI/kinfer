package mlxtts

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// debug logs where a frame's time goes (KINFER_TTS_DEBUG).
var debug = os.Getenv("KINFER_TTS_DEBUG") != ""

// Request is one thing to say.
type Request struct {
	Text     string
	Voice    string // one of Voices
	Language string // one of Languages, or "auto"
	Instruct string // the tone, "用开心的语气说"; "" for none

	Temperature float32 // 0.9 as mlx-audio has it; 0 for the most likely codes
	TopK        int     // 50
	Penalty     float32 // repetition penalty, 1.05
	MaxFrames   int     // 4096 (5½ minutes)

	// Stream is how many frames each piece of a streamed answer holds (12.5
	// a second); 0 decodes the whole once it is made, as mlx-audio's
	// non-streaming path does.
	Stream int
}

// Defaults fills what r leaves out with mlx-audio's defaults.
func (r Request) withDefaults() Request {
	if r.TopK == 0 {
		r.TopK = 50
	}
	if r.Penalty == 0 {
		r.Penalty = 1.05
	}
	if r.MaxFrames <= 0 {
		r.MaxFrames = 4096
	}
	if r.Language == "" {
		r.Language = "auto"
	}
	return r
}

// Result is what was said: the codes and how long each part took.
type Result struct {
	Frames  int
	Samples int
	Codes   [][]int32 // [frames][16]
	Prefill time.Duration
	Talk    time.Duration // the frames after the first
	Decode  time.Duration
	First   time.Duration // to the first audio handed over
	Stopped bool          // the frames ran out before the model ended
}

// inputs is mlx-audio's _prepare_generation_inputs: the prompt's embeddings
// [1, P, hidden], the text read one token a frame after it [1, T, hidden], and
// the pad read once the text is through [1, 1, hidden].
func (m *Model) inputs(r Request) (prompt, trailing, pad mlx.Array, err error) {
	tc := m.cfg.Talker
	t := m.talker
	ids := m.tok.Encode("<|im_start|>assistant\n"+r.Text+"<|im_end|>\n<|im_start|>assistant\n", true)
	if len(ids) < 9 {
		return 0, 0, 0, fmt.Errorf("nothing to say")
	}
	text, err := t.text(ids)
	if err != nil {
		return 0, 0, 0, err
	}
	tts, err := t.text([]int32{m.cfg.TTSBOS, m.cfg.TTSEOS, m.cfg.TTSPad})
	if err != nil {
		return 0, 0, 0, err
	}
	hid := tc.Hidden
	part := func(a mlx.Array, from, to int) mlx.Array { return mlx.Slice(a, []int{0, from, 0}, []int{1, to, hid}) }
	bos, eos := part(tts, 0, 1), part(tts, 1, 2)
	pad = part(tts, 2, 3)

	voice := strings.ToLower(r.Voice)
	spk, ok := tc.Speakers[voice]
	if !ok {
		return 0, 0, 0, fmt.Errorf("voice %q: this model speaks in %s", r.Voice, strings.Join(m.Voices(), ", "))
	}
	lang := strings.ToLower(r.Language)
	langID, haveLang := int32(0), false
	if lang != "auto" {
		if id, ok := tc.Languages[lang]; ok {
			langID, haveLang = id, true
		}
	}
	if d, ok := tc.Dialects[voice].(string); ok && (lang == "chinese" || lang == "auto") {
		if id, ok := tc.Languages[d]; ok {
			langID, haveLang = id, true
		}
	}
	prefill := []int32{tc.NoThink, tc.ThinkBOS, tc.ThinkEOS}
	if haveLang {
		prefill = []int32{tc.Think, tc.ThinkBOS, langID, tc.ThinkEOS}
	}
	codec := mlx.Concatenate(1,
		t.codec(mlx.FromInt32(prefill, 1, len(prefill))),
		t.codec(mlx.FromInt32([]int32{spk}, 1, 1)),
		t.codec(mlx.FromInt32([]int32{tc.CodecPad, tc.CodecBOS}, 1, 2)))
	n := codec.Shape()[1]

	pads := mlx.BroadcastTo(pad, 1, n-2, hid)
	combined := mlx.Add(mlx.Concatenate(1, pads, bos), part(codec, 0, n-1))
	parts := []mlx.Array{}
	if r.Instruct != "" {
		it, err := t.text(m.tok.Encode("<|im_start|>user\n"+r.Instruct+"<|im_end|>\n", true))
		if err != nil {
			return 0, 0, 0, err
		}
		parts = append(parts, it)
	}
	L := len(ids)
	parts = append(parts, part(text, 0, 3), combined, mlx.Add(part(text, 3, 4), part(codec, n-1, n)))
	prompt = mlx.Concatenate(1, parts...)
	trailing = mlx.Concatenate(1, part(text, 4, L-5), eos)
	return prompt, trailing, pad, nil
}

// Speak says r.Text. onAudio is handed the audio, [-1, 1] at 24 kHz: all of
// it at the end, or a piece every r.Stream frames; an error from it stops
// the speech. One call at a time: the model holds one sequence.
func (m *Model) Speak(ctx context.Context, r Request, onAudio func(pcm []float32) error) (res Result, err error) {
	defer mlx.Recover(&err)
	r = r.withDefaults()
	tc := m.cfg.Talker
	start := time.Now()
	if debug {
		mlx.ResetPeak()
	}

	outer := mlx.NewScope()
	defer outer.Close()
	prompt, trailing, pad, err := m.inputs(r)
	if err != nil {
		return res, err
	}
	trail := trailing.Shape()[1]
	cache := newCache(tc.Layers)
	defer cache.free()
	dt := m.talker.main.dt
	sFirst := newSampling(r.Temperature, r.TopK, r.Penalty, tc.CodecEOS, tc.Vocab, dt)
	defer sFirst.free()
	sRest := newSampling(r.Temperature, r.TopK, 1, 0, tc.CodePredictor.Vocab, dt)
	defer sRest.free()

	var st *codecState
	if r.Stream > 0 {
		st = newCodecState(m.codec.cfg.Layers)
		defer st.free()
	}
	decoded := 0
	hand := func(final bool) error {
		want := r.Stream
		if decoded == 0 {
			want = min(want, firstPiece)
		}
		if n := len(res.Codes) - decoded; st == nil || n == 0 || (!final && n < want) {
			return nil
		}
		t0 := time.Now()
		pcm := m.decodePiece(res.Codes[decoded:], st)
		res.Decode += time.Since(t0)
		decoded = len(res.Codes)
		if res.First == 0 {
			res.First = time.Since(start)
		}
		res.Samples += len(pcm)
		return onAudio(pcm)
	}

	// Frames are pipelined: while the GPU works out one, the next one's
	// graph is built on the codes it will have drawn, so the building costs
	// no time. Nothing the next frame reads is waited for — the repetition
	// penalty's recent codes stay on the GPU — and a frame is read only to
	// keep its codes and see whether the model has ended, which leaves one
	// frame built for nothing at the end.
	var buildT, asyncT, waitT time.Duration
	window := mlx.Full(mlx.Int32, -1, recentCodes) // the first codes so far, -1 for none yet
	build := func(f int, in, win mlx.Array) frameGraph {
		tb := time.Now()
		var text mlx.Array
		if f < trail {
			text = mlx.Slice(trailing, []int{0, f, 0}, []int{1, f + 1, tc.Hidden})
		} else {
			text = pad
		}
		L := 1
		if f == 0 {
			L = prompt.Shape()[1]
		}
		g := m.frame(in, L, win, text, cache, sFirst, sRest)
		ta := time.Now()
		mlx.AsyncEval(g.next, g.codes, g.window)
		buildT += ta.Sub(tb)
		asyncT += time.Since(ta)
		return g
	}
	cur := build(0, prompt, window)
	for frame := 0; ; frame++ {
		var next *frameGraph
		if frame+1 < r.MaxFrames && ctx.Err() == nil {
			n := build(frame+1, cur.next, cur.window)
			next = &n
		}
		tw := time.Now()
		got := cur.codes.Int32s()
		waitT += time.Since(tw)
		cur.next.Free() // the next frame's graph holds it
		cur.free()
		if frame == 0 {
			res.Prefill = time.Since(start)
		}
		if got[0] == tc.CodecEOS || next == nil {
			if next != nil {
				next.next.Free()
				next.free()
			} else if got[0] != tc.CodecEOS {
				res.Codes = append(res.Codes, got)
			}
			if err := ctx.Err(); err != nil {
				return res, err
			}
			break
		}
		res.Codes = append(res.Codes, got)
		if err := hand(false); err != nil {
			next.next.Free()
			next.free()
			return res, err
		}
		cur = *next
	}
	res.Frames = len(res.Codes)
	if debug {
		mu := mlx.MemoryUse()
		log.Printf("mlxtts: %d frames: building %v, handing to MLX %v, waiting for the GPU %v; MLX memory %.2f GB active, %.2f GB cached, %.2f GB peak",
			res.Frames, buildT, asyncT, waitT, float64(mu.Active)/1e9, float64(mu.Cache)/1e9, float64(mu.Peak)/1e9)
	}
	res.Stopped = res.Frames == r.MaxFrames
	res.Talk = time.Since(start) - res.Prefill - res.Decode
	if st != nil {
		err = hand(true)
		mlx.ClearCache()
		return res, err
	}
	if res.Frames == 0 {
		mlx.ClearCache()
		return res, nil
	}
	t0 := time.Now()
	pcm := m.decode(res.Codes)
	res.Decode = time.Since(t0)
	res.First = time.Since(start)
	res.Samples = len(pcm)
	mlx.ClearCache()
	return res, onAudio(pcm)
}

// firstPiece is the most frames a streamed answer's first piece waits for:
// 0.32 s of speech, out sooner than a whole piece, while the speech is made
// four times as fast as it plays, so the pieces after it keep ahead.
const firstPiece = 4

// frameGraph is one frame, built and not yet read: the next frame's input,
// this frame's sixteen codes [1, 16] int32, and the recent first codes with
// this frame's. Each is the caller's to free.
type frameGraph struct{ next, codes, window mlx.Array }

func (g frameGraph) free() { g.codes.Free(); g.window.Free() }

// frame builds a frame: the talker reads in (the prompt, or the last
// frame's codes and a text token) and draws the first code, the code
// predictor the other fifteen, and their embeddings and text — the next
// text token, or the pad — make the next input. The cache's arrays are
// replaced and the old ones let go, so that MLX writes the new keys and
// values into their buffers rather than into copies.
func (m *Model) frame(in mlx.Array, L int, window, text mlx.Array, cache *kvCache, sFirst, sRest *sampling) frameGraph {
	t := m.talker
	tc := m.cfg.Talker
	sc := mlx.NewScope()
	defer sc.Close()
	old := cache.arrays()
	logits, hidden := t.step(in, L, cache)
	first := sFirst.draw(sFirst.penalize(logits, window))
	codes := []mlx.Array{first}
	pc := newCache(tc.CodePredictor.Layers)
	pc.step = tc.CodePredictor.Groups + 1 // a frame's codes: the buffers never grow
	for g := 0; g < tc.CodePredictor.Groups-1; g++ {
		x, n := mlx.Array(0), 1
		if g == 0 {
			x, n = mlx.Concatenate(1, hidden, t.codec(first)), 2
		} else {
			x = rows(t.cpEmbed[g-1], codes[len(codes)-1])
		}
		codes = append(codes, sRest.draw(t.predict(x, n, pc, g)))
	}
	next := t.codec(first)
	for g, c := range codes[1:] {
		next = mlx.Add(next, rows(t.cpEmbed[g], c))
	}
	next = mlx.Add(text, next)
	all := mlx.Astype(mlx.Concatenate(1, codes...), mlx.Int32)
	win := mlx.Concatenate(0, mlx.Slice(window, []int{1}, []int{recentCodes}), mlx.Reshape(mlx.Astype(first, mlx.Int32), 1))
	sc.Escape(next, all, win)
	sc.Escape(cache.arrays()...)
	for _, a := range old {
		a.Free()
	}
	return frameGraph{next: next, codes: all, window: win}
}

// decodePieceFrames is how many frames a whole decode reads at a time. The
// codec is causal and carries its state from piece to piece, so the audio is
// the audio of one pass — mlx-audio's, up to its 300-frame chunks, beyond
// which it reads each chunk after only 25 frames of the last — while what is
// held at once stays small: a 2,000-second pass of the last block's
// convolutions takes gigabytes.
const decodePieceFrames = 50

// decode is the audio of codes as mlx-audio's speech_tokenizer.decode makes
// it, trimmed as it trims: to 1,920 samples for each frame whose first code
// is above 0.
func (m *Model) decode(codes [][]int32) []float32 {
	var pcm []float32
	st := newCodecState(m.codec.cfg.Layers)
	defer st.free()
	for start := 0; start < len(codes); start += decodePieceFrames {
		pcm = append(pcm, m.decodePiece(codes[start:min(start+decodePieceFrames, len(codes))], st)...)
	}
	valid := 0
	for _, c := range codes {
		if c[0] > 0 {
			valid++
		}
	}
	if n := valid * m.codec.perFrame; n > 0 && n < len(pcm) {
		pcm = pcm[:n]
	}
	return pcm
}

// decodePiece runs the codec over codes, continuing st, and returns the
// audio; st keeps what the next piece needs.
//
// The graph's intermediates are let go before it is evaluated, so that MLX
// frees each once it has been read rather than holding them all.
func (m *Model) decodePiece(codes [][]int32, st *codecState) []float32 {
	flat := make([]int32, 0, len(codes)*len(codes[0]))
	for _, c := range codes {
		flat = append(flat, c...)
	}
	old := st.arrays()
	sc := mlx.NewScope()
	wav := m.codec.step(mlx.FromInt32(flat, 1, len(codes), len(codes[0])), st)
	keep := st.arrays()
	sc.Escape(wav)
	sc.Escape(keep...)
	sc.Close()
	stale := map[mlx.Array]bool{}
	for _, a := range old {
		stale[a] = true
	}
	for _, a := range keep {
		delete(stale, a)
	}
	for a := range stale {
		a.Free()
	}
	mlx.Eval(append([]mlx.Array{wav}, keep...)...)
	defer wav.Free()
	return wav.Float32s()
}
