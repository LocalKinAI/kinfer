package mlximage

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/mlxengine"
)

// readNpy reads a little-endian float32 .npy file as numpy saves one.
func readNpy(t *testing.T, path string) ([]float32, []int) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	hl := int(binary.LittleEndian.Uint16(b[8:10]))
	hdr := string(b[10 : 10+hl])
	if !strings.Contains(hdr, "'<f4'") {
		t.Fatalf("%s is not float32: %s", path, hdr)
	}
	var shape []int
	if m := regexp.MustCompile(`\(([^)]*)\)`).FindStringSubmatch(hdr); m != nil {
		for _, s := range strings.Split(m[1], ",") {
			if s = strings.TrimSpace(s); s != "" {
				v, _ := strconv.Atoi(s)
				shape = append(shape, v)
			}
		}
	}
	data := b[10+hl:]
	out := make([]float32, len(data)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out, shape
}

func compare(a, b []float32) (maxd, cos float64) {
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		maxd = math.Max(maxd, math.Abs(x-y))
		dot += x * y
		na += x * x
		nb += y * y
	}
	return maxd, dot / math.Sqrt(na*nb)
}

func f32(a mlx.Array) []float32 { return mlx.Astype(a, mlx.Float32).Float32s() }

// With KINFER_IMAGE_MODEL (mflux's Z-Image-Turbo directory) and
// KINFER_IMAGE_REF (zimage_ref.py's output), each part is checked against
// mflux's, run eagerly: the tokens, the caption's features, the schedule, the
// noise, the first step's velocity and latents (fed mflux's caption and
// noise), the decoder on mflux's latents, and the whole picture.
func TestAgainstMflux(t *testing.T) {
	dir, ref := os.Getenv("KINFER_IMAGE_MODEL"), os.Getenv("KINFER_IMAGE_REF")
	if dir == "" || ref == "" {
		t.Skip("KINFER_IMAGE_MODEL and KINFER_IMAGE_REF not set")
	}
	libs, err := mlxengine.Libraries()
	if err != nil {
		t.Fatal(err)
	}
	if err := mlx.Open(libs); err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	t.Logf("loaded in %v, %.2f GB", time.Since(t0).Round(time.Millisecond), float64(m.Bytes())/1e9)

	var r struct {
		Prompt        string
		Seed          uint64
		Width, Height int
		IDs           []int32 `json:"ids"`
	}
	b, _ := os.ReadFile(filepath.Join(ref, "ref.json"))
	json.Unmarshal(b, &r)
	load := func(name string) ([]float32, []int) { return readNpy(t, filepath.Join(ref, name+".npy")) }

	ids := m.promptIDs(r.Prompt)
	if fmt.Sprint(ids) != fmt.Sprint(r.IDs) {
		t.Errorf("tokens %v, mflux %v", ids, r.IDs)
	}
	capRef, capShape := load("cap")
	cap := m.te.encode(ids)
	d, c := compare(f32(cap), capRef)
	t.Logf("caption %v: max |Δ| %.3g cos %.7f", capShape, d, c)

	sigRef, _ := load("sigmas")
	sig := f32(sigmas(9, r.Width, r.Height))
	d, _ = compare(sig, sigRef)
	t.Logf("sigmas %v (mflux %v) max |Δ| %.3g", sig, sigRef, d)
	if d != 0 {
		t.Errorf("the schedule differs")
	}

	noiseRef, ns := load("noise")
	noise := mlx.Astype(mlx.RandomNormal(mlx.RandomKey(r.Seed), mlx.Float32, latentC, 1, r.Height/8, r.Width/8), mlx.Bfloat16)
	d, _ = compare(f32(noise), noiseRef)
	t.Logf("noise %v max |Δ| %.3g", ns, d)
	if d != 0 {
		t.Errorf("the noise differs")
	}

	// The first step, on mflux's caption and noise.
	capM := mlx.Astype(mlx.FromFloat32(capRef, capShape...), mlx.Bfloat16)
	x := mlx.Astype(mlx.FromFloat32(noiseRef, ns...), mlx.Bfloat16)
	sigA := mlx.FromFloat32(sigRef, len(sigRef))
	st := mlx.Slice(sigA, []int{0}, []int{1})
	v := m.dit.forward(x, mlx.Subtract(mlx.Ones(mlx.Float32, 1), st), capM)
	vRef, _ := load("velocity0")
	d, c = compare(f32(v), vRef)
	t.Logf("first velocity: max |Δ| %.3g cos %.7f", d, c)
	if c < 0.999 {
		t.Errorf("the first velocity is further from mflux's than rounding")
	}
	dt := mlx.Astype(mlx.Subtract(mlx.Slice(sigA, []int{1}, []int{2}), st), x.Dtype())
	x1 := mlx.Add(x, mlx.Multiply(mlx.Astype(v, x.Dtype()), dt))
	l1Ref, _ := load("latents1")
	d, c = compare(f32(x1), l1Ref)
	t.Logf("latents after one step: max |Δ| %.3g cos %.7f", d, c)

	// The decoder on mflux's latents.
	latRef, ls := load("latents")
	img := m.vae.decode(mlx.Astype(mlx.FromFloat32(latRef, ls...), mlx.Bfloat16))
	two := mlx.ScalarAs(2, img.Dtype())
	img = mlx.Minimum(mlx.Maximum(mlx.AddScalar(mlx.Divide(img, two), 0.5), mlx.ScalarAs(0, img.Dtype())), mlx.ScalarAs(1, img.Dtype()))
	imgRef, _ := load("image") // [1, 3, 1, H, W]
	got := f32(mlx.Contiguous(mlx.Transpose(img, 0, 3, 1, 2)))
	d, c = compare(got, imgRef)
	t.Logf("decoder on mflux's latents: max |Δ| %.3g cos %.7f", d, c)
	if c < 0.999 {
		t.Errorf("the decoder is further from mflux's than rounding")
	}

	// The whole loop, kinfer's own way, against mflux's eager one.
	seed := r.Seed
	_, lat := m.pixels(Request{Prompt: r.Prompt, Width: r.Width, Height: r.Height, Seed: &seed})
	d, c = compare(f32(lat), latRef)
	t.Logf("latents after 9 steps, kinfer's caption and loop: max |Δ| %.3g cos %.7f", d, c)

	// The whole picture, twice: the second is the one timed warm.
	if _, err := m.Paint(context.Background(), Request{Prompt: r.Prompt, Width: r.Width, Height: r.Height, Seed: &seed}); err != nil {
		t.Fatal(err)
	}
	res, err := m.Paint(context.Background(), Request{Prompt: r.Prompt, Width: r.Width, Height: r.Height, Seed: &seed})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("painted %d×%d in %v: encode %v, denoise %v, decode %v", res.Width, res.Height, res.Total.Round(time.Millisecond),
		res.Encode.Round(time.Millisecond), res.Denoise.Round(time.Millisecond), res.Decode.Round(time.Millisecond))
	out := filepath.Join(ref, "kinfer.png")
	os.WriteFile(out, res.PNG, 0o644)
	if f, err := os.Open(filepath.Join(ref, "mflux.png")); err == nil {
		mp, _ := png.Decode(f)
		f.Close()
		kp, _ := png.Decode(strings.NewReader(string(res.PNG)))
		var se float64
		bnd := mp.Bounds()
		for y := bnd.Min.Y; y < bnd.Max.Y; y++ {
			for xx := bnd.Min.X; xx < bnd.Max.X; xx++ {
				r1, g1, b1, _ := mp.At(xx, y).RGBA()
				r2, g2, b2, _ := kp.At(xx, y).RGBA()
				for _, p := range [][2]uint32{{r1, r2}, {g1, g2}, {b1, b2}} {
					dd := float64(p[0]>>8) - float64(p[1]>>8)
					se += dd * dd
				}
			}
		}
		mse := se / float64(bnd.Dx()*bnd.Dy()*3)
		t.Logf("against mflux's own picture: PSNR %.1f dB (%s)", 10*math.Log10(255*255/mse), out)
	}
}

// With KINFER_IMAGE_MODEL and KINFER_IMAGE_MEMORY, paints three 1024² pictures
// in a row under the runner's cache limit and says what MLX holds after each:
// it must not grow from one to the next.
func TestPaintMemory(t *testing.T) {
	dir := os.Getenv("KINFER_IMAGE_MODEL")
	if dir == "" || os.Getenv("KINFER_IMAGE_MEMORY") == "" {
		t.Skip("KINFER_IMAGE_MODEL and KINFER_IMAGE_MEMORY not set")
	}
	libs, err := mlxengine.Libraries()
	if err != nil {
		t.Fatal(err)
	}
	if err := mlx.Open(libs); err != nil {
		t.Fatal(err)
	}
	mlx.SetCacheLimit(cacheLimit)
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	var held []float64
	for i, size := range [][2]int{{1024, 1024}, {1024, 768}, {1024, 768}} {
		seed := uint64(7)
		mlx.ResetPeak()
		res, err := m.Paint(context.Background(), Request{Prompt: "a lighthouse on a rocky coast at sunset, oil painting", Width: size[0], Height: size[1], Seed: &seed})
		if err != nil {
			t.Fatal(err)
		}
		u := mlx.MemoryUse()
		held = append(held, float64(u.Active+u.Cache)/1e9)
		t.Logf("%d: %d×%d in %v (text %v, denoise %v, decode %v); MLX active %.2f GB, cached %.2f GB, peak %.2f GB", i+1, size[0], size[1],
			res.Total.Round(time.Millisecond), res.Encode.Round(time.Millisecond), res.Denoise.Round(time.Millisecond), res.Decode.Round(time.Millisecond),
			float64(u.Active)/1e9, float64(u.Cache)/1e9, float64(u.Peak)/1e9)
	}
	if held[2] > held[1]+0.5 {
		t.Errorf("MLX holds more after each picture: %v GB", held)
	}
}

// With KINFER_IMAGE_QMM, times MLX's 8-bit quantized matmul at the
// transformer's shapes for a 1024² picture (4128 tokens), to set against the
// same in Python's MLX.
func TestQuantizedMatmulSpeed(t *testing.T) {
	if os.Getenv("KINFER_IMAGE_QMM") == "" {
		t.Skip("KINFER_IMAGE_QMM not set")
	}
	libs, err := mlxengine.Libraries()
	if err != nil {
		t.Fatal(err)
	}
	if err := mlx.Open(libs); err != nil {
		t.Fatal(err)
	}
	for _, s := range [][3]int{{4128, 3840, 3840}, {4128, 3840, 10240}, {4128, 10240, 3840}, {1056, 3840, 3840}} {
		M, K, N := s[0], s[1], s[2]
		key := mlx.RandomKey(1)
		x := mlx.Astype(mlx.RandomNormal(key, mlx.Float32, M, K), mlx.Bfloat16)
		w := mlx.Astype(mlx.RandomNormal(key, mlx.Float32, N, K), mlx.Bfloat16)
		// Quantize through the dequantize round trip's inverse is not bound;
		// time a plain matmul and the model's own quantized weights instead.
		mlx.Eval(x, w)
		run := func(f func() mlx.Array) time.Duration {
			for i := 0; i < 3; i++ {
				mlx.Eval(f())
			}
			t0 := time.Now()
			for i := 0; i < 20; i++ {
				mlx.Eval(f())
			}
			return time.Since(t0) / 20
		}
		mm := run(func() mlx.Array { return mlx.Matmul(x, mlx.T(w)) })
		t.Logf("bf16 matmul %dx%d·%dx%d: %v", M, K, K, N, mm)
	}
	dir := os.Getenv("KINFER_IMAGE_MODEL")
	ws, err := loadWeights(dir, "transformer")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"layers.0.attention.to_q", "layers.0.feed_forward.w1", "layers.0.feed_forward.w2"} {
		l := ws.lin(name)
		K := l.W.Shape()[1] * 32 / l.Bits
		for _, M := range []int{4128, 1056} {
			x := mlx.Astype(mlx.RandomNormal(mlx.RandomKey(2), mlx.Float32, 1, M, K), mlx.Bfloat16)
			mlx.Eval(x)
			for i := 0; i < 3; i++ {
				mlx.Eval(l.apply(x))
			}
			t0 := time.Now()
			for i := 0; i < 20; i++ {
				mlx.Eval(l.apply(x))
			}
			t.Logf("q%d %s, %d tokens: %v (group %d)", l.Bits, name, M, time.Since(t0)/20, l.Group)
		}
	}
}

// With KINFER_IMAGE_QMM, times the transformer's attention and one block at
// a 1024² picture's 4128 tokens.
func TestAttentionSpeed(t *testing.T) {
	if os.Getenv("KINFER_IMAGE_QMM") == "" {
		t.Skip("KINFER_IMAGE_QMM not set")
	}
	libs, _ := mlxengine.Libraries()
	if err := mlx.Open(libs); err != nil {
		t.Fatal(err)
	}
	const L = 4128
	key := mlx.RandomKey(3)
	q := mlx.Astype(mlx.RandomNormal(key, mlx.Float32, 1, 30, L, 128), mlx.Bfloat16)
	mask := mlx.Zeros(mlx.Bfloat16, 1, 1, 1, L)
	mlx.Eval(q, mask)
	timeIt := func(name string, f func() mlx.Array) {
		for i := 0; i < 3; i++ {
			mlx.Eval(f())
		}
		t0 := time.Now()
		for i := 0; i < 20; i++ {
			mlx.Eval(f())
		}
		t.Logf("%s: %v", name, time.Since(t0)/20)
	}
	timeIt("sdpa, mask", func() mlx.Array { return mlx.SDPA(q, q, q, 0.088, "", mask) })
	timeIt("sdpa, none", func() mlx.Array { return mlx.SDPA(q, q, q, 0.088, "", 0) })
	x := mlx.Astype(mlx.RandomNormal(key, mlx.Float32, 1, L, 30, 128), mlx.Bfloat16)
	f := mlx.RandomNormal(key, mlx.Float32, L, 64, 2)
	mlx.Eval(x, f)
	timeIt("rotary", func() mlx.Array { return rotary(x, f) })
	dir := os.Getenv("KINFER_IMAGE_MODEL")
	ws, err := loadWeights(dir, "transformer")
	if err != nil {
		t.Fatal(err)
	}
	b := loadBlock(ws, "layers.0.", true)
	h := mlx.Astype(mlx.RandomNormal(key, mlx.Float32, 1, L, 3840), mlx.Bfloat16)
	temb := mlx.Astype(mlx.RandomNormal(key, mlx.Float32, 1, 256), mlx.Bfloat16)
	mlx.Eval(h, temb)
	timeIt("one block", func() mlx.Array { return b.forward(h, mask, f, temb) })
}
