package mlximage

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/png"
	"log"
	"math"
	"os"
	"time"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// debug logs each step's time (KINFER_IMAGE_DEBUG).
var debug = os.Getenv("KINFER_IMAGE_DEBUG") != ""

// Request is a picture to make.
type Request struct {
	Prompt        string
	Width, Height int     // pixels, multiples of 16; 1024 each when 0
	Steps         int     // 9 when 0, mflux's default for Z-Image-Turbo
	Seed          *uint64 // the same seed, the same picture; drawn from the clock when nil
}

// Result is what a picture cost.
type Result struct {
	PNG             []byte
	Width, Height   int
	Seed            uint64
	Tokens          int
	Encode, Denoise time.Duration
	Decode, Total   time.Duration
}

func (r Request) withDefaults() Request {
	if r.Width == 0 {
		r.Width = 1024
	}
	if r.Height == 0 {
		r.Height = 1024
	}
	if r.Steps == 0 {
		r.Steps = 9
	}
	return r
}

// The sizes a picture may have: multiples of 16 (the VAE's 8 by the
// transformer's 2×2 patches), from 256 to 2048 a side.
func checkSize(w, h int) error {
	for _, v := range []int{w, h} {
		if v%16 != 0 || v < 256 || v > 2048 {
			return fmt.Errorf("a picture is 256 to 2048 pixels a side, in multiples of 16, not %d×%d", w, h)
		}
	}
	return nil
}

// sigmas is LinearScheduler's schedule for steps at w×h, shifted as
// Z-Image-Turbo's requires_sigma_shift has it, then 0: float32 [steps+1].
func sigmas(steps, w, h int) mlx.Array {
	// mx.linspace(1, 1/steps, steps): MLX interpolates between the ends.
	t := mlx.Divide(mlx.Arange(0, float64(steps), 1, mlx.Float32), mlx.Scalar(float32(steps-1)))
	base := mlx.Add(mlx.Multiply(mlx.Subtract(mlx.Scalar(1), t), mlx.Scalar(1)), mlx.Multiply(t, mlx.Scalar(float32(1/float64(steps)))))
	m := (1.15 - 0.5) / float64(4096-256)
	b := 0.5 - m*256
	mu := mlx.Scalar(float32(m*float64(w*h)/256 + b))
	e := mlx.Exp(mu)
	shifted := mlx.Divide(e, mlx.Add(e, mlx.Subtract(mlx.Divide(mlx.Scalar(1), base), mlx.Scalar(1))))
	return mlx.Concatenate(0, shifted, mlx.Zeros(mlx.Float32, 1))
}

// Paint makes the picture r asks for. One call at a time.
func (m *Model) Paint(ctx context.Context, r Request) (res Result, err error) {
	defer mlx.Recover(&err)
	r = r.withDefaults()
	if err := checkSize(r.Width, r.Height); err != nil {
		return res, err
	}
	start := time.Now()
	res.Width, res.Height = r.Width, r.Height
	res.Seed = uint64(time.Now().UnixNano())
	if r.Seed != nil {
		res.Seed = *r.Seed
	}

	outer := mlx.NewScope()
	defer outer.Close()
	ids := m.promptIDs(r.Prompt)
	res.Tokens = len(ids)
	cap := evaluated(func() mlx.Array { return m.te.encode(ids) })
	defer cap.Free()
	mlx.ClearCache()
	res.Encode = time.Since(start)

	t0 := time.Now()
	key := mlx.RandomKey(res.Seed)
	x := mlx.Astype(mlx.RandomNormal(key, mlx.Float32, latentC, 1, r.Height/8, r.Width/8), mlx.Bfloat16)
	sig := sigmas(r.Steps, r.Width, r.Height)
	mlx.Eval(x, sig)
	own := false // whether x is one of the steps', which this loop frees
	for t := 0; t < r.Steps; t++ {
		if err := ctx.Err(); err != nil {
			if own {
				x.Free()
			}
			return res, err
		}
		next := evaluated(func() mlx.Array { return m.step(x, sig, t, cap) })
		if own {
			x.Free()
		}
		x, own = next, true
		if debug {
			log.Printf("image: step %d of %d at %v", t+1, r.Steps, time.Since(t0).Round(time.Millisecond))
		}
	}
	res.Denoise = time.Since(t0)

	t0 = time.Now()
	img := evaluated(func() mlx.Array {
		img := m.vae.decode(x)
		// ImageUtil.to_image: clip(x/2 + 0.5, 0, 1) in the decoder's dtype,
		// then float32, ×255, rounded.
		dt := img.Dtype()
		img = mlx.Minimum(mlx.Maximum(mlx.AddScalar(mlx.Divide(img, mlx.ScalarAs(2, dt)), 0.5), mlx.ScalarAs(0, dt)), mlx.ScalarAs(1, dt))
		return mlx.Astype(img, mlx.Float32)
	})
	px := img.Float32s()
	img.Free()
	if own {
		x.Free()
	}
	res.Decode = time.Since(t0)
	mlx.ClearCache()
	if debug {
		u := mlx.MemoryUse()
		log.Printf("image: after the picture, MLX holds %.2f GB active, %.2f GB cached, peak %.2f GB",
			float64(u.Active)/1e9, float64(u.Cache)/1e9, float64(u.Peak)/1e9)
	}

	pic := image.NewRGBA(image.Rect(0, 0, r.Width, r.Height))
	for i := 0; i < r.Width*r.Height; i++ {
		for c := 0; c < 3; c++ {
			pic.Pix[i*4+c] = uint8(math.RoundToEven(float64(px[i*3+c] * 255))) // numpy's float32 product, rounded half to even
		}
		pic.Pix[i*4+3] = 255
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, pic); err != nil {
		return res, err
	}
	res.PNG = buf.Bytes()
	res.Total = time.Since(start)
	return res, nil
}

// pixels is Paint up to the decoder's output, for the reference test: the
// picture [1, h, w, 3] in [-1, 1] and the latents it came from.
func (m *Model) pixels(r Request) (img, latents mlx.Array) {
	r = r.withDefaults()
	ids := m.promptIDs(r.Prompt)
	cap := evaluated(func() mlx.Array { return m.te.encode(ids) })
	x := mlx.Astype(mlx.RandomNormal(mlx.RandomKey(*r.Seed), mlx.Float32, latentC, 1, r.Height/8, r.Width/8), mlx.Bfloat16)
	sig := sigmas(r.Steps, r.Width, r.Height)
	mlx.Eval(x, sig)
	for t := 0; t < r.Steps; t++ {
		x = evaluated(func() mlx.Array { return m.step(x, sig, t, cap) })
	}
	return evaluated(func() mlx.Array { return m.vae.decode(x) }), x
}

// step is one denoising step from latents x: the velocity at sigma[t], and
// the Euler step to sigma[t+1], as LinearScheduler.step takes it.
func (m *Model) step(x, sig mlx.Array, t int, cap mlx.Array) mlx.Array {
	st := mlx.Slice(sig, []int{t}, []int{t + 1})
	noise := m.dit.forward(x, mlx.Subtract(mlx.Ones(mlx.Float32, 1), st), cap)
	dt := mlx.Astype(mlx.Subtract(mlx.Slice(sig, []int{t + 1}, []int{t + 2}), st), x.Dtype())
	return mlx.Add(x, mlx.Multiply(mlx.Astype(noise, x.Dtype()), dt))
}

// evaluated builds what build returns in a scope of its own, frees every
// intermediate's handle, then evaluates it: MLX frees each intermediate's
// buffer as soon as the graph is done with it, where a handle still held
// would keep every one of them to the end — a 1024² picture's attention, tens
// of gigabytes. The caller frees the result.
func evaluated(build func() mlx.Array) mlx.Array {
	t0 := time.Now()
	sc := mlx.NewScope()
	a := build()
	t1 := time.Now()
	sc.Escape(a)
	sc.Close()
	t2 := time.Now()
	mlx.Eval(a)
	if debug {
		log.Printf("image: built in %v, freed in %v, evaluated in %v", t1.Sub(t0).Round(time.Microsecond), t2.Sub(t1).Round(time.Microsecond), time.Since(t2).Round(time.Millisecond))
	}
	return a
}
