package mlxembed

import (
	"image"
	"image/color"
	"math"
)

// resizeRGB scales img to w×h with a Catmull-Rom kernel, as packed RGB, its
// alpha dropped rather than composited (the reference converts to RGB before
// it resizes). It follows golang.org/x/image/draw's Kernel.Scale, which Ollama
// resizes with: the kernel widened by the factor when shrinking, each output
// pixel's weights normalized, rows first into a float buffer, then columns,
// on 16-bit premultiplied values rounded back to 8 bits at the end.
func resizeRGB(img image.Image, w, h int) []uint8 {
	b := img.Bounds()
	sw, sh := b.Dx(), b.Dy()
	opaque := isOpaque(img)
	at := func(x, y int) (r, g, bl, a uint32) {
		c := img.At(b.Min.X+x, b.Min.Y+y)
		if !opaque {
			n := color.NRGBAModel.Convert(c).(color.NRGBA)
			c = color.NRGBA{n.R, n.G, n.B, 0xff}
		}
		return c.RGBA()
	}

	hz, vt := newDistrib(w, sw), newDistrib(h, sh)
	tmp := make([][4]float64, sh*w)
	for y := 0; y < sh; y++ {
		for x, s := range hz.sources {
			var p [4]float64
			for _, c := range hz.contribs[s.i:s.j] {
				r, g, bl, a := at(c.coord, y)
				p[0] += float64(r) * c.weight
				p[1] += float64(g) * c.weight
				p[2] += float64(bl) * c.weight
				p[3] += float64(a) * c.weight
			}
			for k := range p {
				p[k] *= s.invTotalFFFF
			}
			tmp[y*w+x] = p
		}
	}
	out := make([]uint8, w*h*3)
	for x := 0; x < w; x++ {
		for y, s := range vt.sources {
			var p [4]float64
			for _, c := range vt.contribs[s.i:s.j] {
				q := &tmp[c.coord*w+x]
				for k := range p {
					p[k] += q[k] * c.weight
				}
			}
			for k := 0; k < 3; k++ {
				p[k] = min(p[k], p[3])
			}
			o := (y*w + x) * 3
			out[o] = uint8(ftou(p[0]*s.invTotal) >> 8)
			out[o+1] = uint8(ftou(p[1]*s.invTotal) >> 8)
			out[o+2] = uint8(ftou(p[2]*s.invTotal) >> 8)
		}
	}
	return out
}

func isOpaque(img image.Image) bool {
	o, ok := img.(interface{ Opaque() bool })
	return ok && o.Opaque()
}

func catmullRom(t float64) float64 {
	if t < 1 {
		return (1.5*t-2.5)*t*t + 1
	}
	return ((-0.5*t+2.5)*t-4)*t + 2
}

const catmullRomSupport = 2

type contrib struct {
	coord  int
	weight float64
}

// source is one output coordinate's run in contribs.
type source struct {
	i, j                   int
	invTotal, invTotalFFFF float64
}

type distrib struct {
	sources  []source
	contribs []contrib
}

// newDistrib is the weights from sw source coordinates to each of dw.
func newDistrib(dw, sw int) distrib {
	scale := float64(sw) / float64(dw)
	half, argScale := float64(catmullRomSupport), 1.0
	if scale > 1 {
		half *= scale
		argScale = 1 / scale
	}
	d := distrib{sources: make([]source, dw)}
	for x := range d.sources {
		center := (float64(x)+0.5)*scale - 0.5
		i := max(int(math.Floor(center-half)), 0)
		j := int(math.Ceil(center + half))
		if j > sw {
			j = max(sw, i)
		}
		total := 0.0
		start := len(d.contribs)
		for c := i; c < j; c++ {
			t := math.Abs((center - float64(c)) * argScale)
			if t >= catmullRomSupport {
				continue
			}
			if wt := catmullRom(t); wt != 0 {
				total += wt
				d.contribs = append(d.contribs, contrib{c, wt})
			}
		}
		total = 1 / total
		d.sources[x] = source{i: start, j: len(d.contribs), invTotal: total, invTotalFFFF: total / 0xffff}
	}
	return d
}

func ftou(f float64) uint16 {
	i := int32(0xffff*f + 0.5)
	if i > 0xffff {
		return 0xffff
	}
	if i > 0 {
		return uint16(i)
	}
	return 0
}
