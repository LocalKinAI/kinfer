package mlximage

import (
	"fmt"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// The VAE's decoder, Flux's (16 latent channels, 8× up), as mflux's
// ZImage VAE: kept in NHWC throughout, where mflux turns each block's input
// to NHWC and its output back, which changes no number.
const (
	vaeScaling = 0.3611
	vaeShift   = 0.1159
	vaeGroups  = 32
	vaeEps     = 1e-6
)

type conv struct{ W, B mlx.Array }

func (c conv) apply(x mlx.Array, pad int) mlx.Array {
	return mlx.Add(mlx.Conv2d(x, c.W, 1, pad), c.B)
}

type groupNorm struct{ W, B mlx.Array }

// apply is nn.GroupNorm with pytorch_compatible: each group's (H·W·C/32)
// values normalized together, by fast.layer_norm, then weight·x + bias.
func (g groupNorm) apply(x mlx.Array) mlx.Array {
	s := x.Shape()
	B, C := s[0], s[3]
	gs := C / vaeGroups
	y := mlx.Reshape(x, B, -1, vaeGroups, gs)
	y = mlx.Reshape(mlx.Transpose(y, 0, 2, 1, 3), B, vaeGroups, -1)
	y = mlx.LayerNorm(y, 0, 0, vaeEps)
	y = mlx.Reshape(y, B, vaeGroups, -1, gs)
	y = mlx.Reshape(mlx.Transpose(y, 0, 2, 1, 3), s...)
	return mlx.Add(mlx.Multiply(g.W, y), g.B)
}

type resnet struct {
	n1, n2       groupNorm
	c1, c2       conv
	shortcut     conv
	haveShortcut bool
}

func (r resnet) apply(x mlx.Array) mlx.Array {
	h := r.c1.apply(mlx.SiLU(r.n1.apply(x)), 1)
	h = r.c2.apply(mlx.SiLU(r.n2.apply(h)), 1)
	if r.haveShortcut {
		x = r.shortcut.apply(x, 0)
	}
	return mlx.Add(x, h)
}

type vaeAttention struct {
	norm       groupNorm
	q, k, v, o linear
	scale      float32
}

// apply is the mid block's self-attention over every position, one head of
// 512, its group norm in float32 as mflux computes it.
func (a vaeAttention) apply(x mlx.Array) mlx.Array {
	s := x.Shape()
	B, H, W, C := s[0], s[1], s[2], s[3]
	h := mlx.Astype(a.norm.apply(mlx.Astype(x, mlx.Float32)), mlx.Bfloat16)
	head := func(t mlx.Array) mlx.Array { return mlx.Transpose(mlx.Reshape(t, B, H*W, 1, C), 0, 2, 1, 3) }
	o := mlx.SDPA(head(a.q.apply(h)), head(a.k.apply(h)), head(a.v.apply(h)), a.scale, "", 0)
	o = mlx.Reshape(mlx.Transpose(o, 0, 2, 1, 3), B, H, W, C)
	return mlx.Add(x, a.o.apply(o))
}

type upBlock struct {
	resnets  []resnet
	up       conv
	upsample bool
}

type decoder struct {
	convIn, convOut conv
	mid             [2]resnet
	attn            vaeAttention
	ups             []upBlock
	normOut         groupNorm
}

func loadDecoder(w *weights) *decoder {
	cv := func(p string) conv { return conv{W: w.get(p + ".weight"), B: w.get(p + ".bias")} }
	gn := func(p string) groupNorm { return groupNorm{W: w.get(p + ".weight"), B: w.get(p + ".bias")} }
	rn := func(p string) resnet {
		r := resnet{n1: gn(p + "norm1"), n2: gn(p + "norm2"), c1: cv(p + "conv1"), c2: cv(p + "conv2")}
		if _, ok := w.t[p+"conv_shortcut.weight"]; ok {
			r.shortcut, r.haveShortcut = cv(p+"conv_shortcut"), true
		}
		return r
	}
	d := &decoder{convIn: cv("decoder.conv_in.conv"), convOut: cv("decoder.conv_out.conv"), normOut: gn("decoder.conv_norm_out.norm")}
	d.mid[0], d.mid[1] = rn("decoder.mid_block.resnets.0."), rn("decoder.mid_block.resnets.1.")
	p := "decoder.mid_block.attentions.0."
	d.attn = vaeAttention{norm: gn(p + "group_norm"), q: w.lin(p + "to_q"), k: w.lin(p + "to_k"), v: w.lin(p + "to_v"), o: w.lin(p + "to_out.0")}
	// mflux's scale: 1 / sqrt(512), each in bfloat16 as its queries are.
	sc := mlx.Divide(mlx.ScalarAs(1, mlx.Bfloat16), mlx.Sqrt(mlx.ScalarAs(512, mlx.Bfloat16)))
	d.attn.scale = mlx.Astype(sc, mlx.Float32).Float32s()[0]
	for i := 0; i < 4; i++ {
		var b upBlock
		for j := 0; j < 3; j++ {
			b.resnets = append(b.resnets, rn(fmt.Sprintf("decoder.up_blocks.%d.resnets.%d.", i, j)))
		}
		if i < 3 {
			b.up, b.upsample = cv(fmt.Sprintf("decoder.up_blocks.%d.upsamplers.0.conv", i)), true
		}
		d.ups = append(d.ups, b)
	}
	return d
}

// upsample2 is nearest-neighbour ×2 of x [B, H, W, C].
func upsample2(x mlx.Array) mlx.Array {
	s := x.Shape()
	B, H, W, C := s[0], s[1], s[2], s[3]
	y := mlx.BroadcastTo(mlx.Reshape(x, B, H, 1, W, 1, C), B, H, 2, W, 2, C)
	return mlx.Reshape(y, B, H*2, W*2, C)
}

// decode turns latents [16, 1, h, w] into the picture [1, 8h, 8w, 3] in
// [-1, 1], bfloat16.
func (d *decoder) decode(latents mlx.Array) mlx.Array {
	s := latents.Shape()
	z := mlx.Reshape(latents, 1, s[0], s[2], s[3]) // ZImageLatentCreator.unpack_latents
	z = mlx.AddScalar(mlx.Divide(z, mlx.ScalarAs(vaeScaling, z.Dtype())), vaeShift)
	x := d.convIn.apply(mlx.Transpose(z, 0, 2, 3, 1), 1)
	x = d.mid[0].apply(x)
	x = d.attn.apply(x)
	x = d.mid[1].apply(x)
	for _, b := range d.ups {
		for _, r := range b.resnets {
			x = r.apply(x)
		}
		if b.upsample {
			x = b.up.apply(upsample2(x), 1)
		}
	}
	x = mlx.Astype(d.normOut.apply(mlx.Astype(x, mlx.Float32)), mlx.Bfloat16)
	return d.convOut.apply(mlx.SiLU(x), 1)
}
