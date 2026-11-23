package mlxengine

import (
	"fmt"
	"hash/fnv"
	"image"
	"image/color"
	"math"

	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/mlxmodel"
	"github.com/LocalKinAI/kinfer/internal/picture"
)

// inkPictureTokens is the most tokens a picture is read as. Inkling 2 takes a
// picture at its own size, a token for every 40 × 40 pixels, and its tower
// weighs every patch against every other: a phone's 12 million pixels would
// be 7,500 tokens and tens of gigabytes inside the tower. A larger picture is
// made smaller first, to about 1,280 × 1,280 or the same area in its own
// shape. Ollama sends any size on as it is.
const inkPictureTokens = 1024

// inkPicture is a picture in a prompt.
type inkPicture struct {
	at, n      int // its tokens in the prompt
	rows, cols int // tiles
	pixels     []float32
	features   mlx.Array // what the tower made of it, once it has
}

// inkPictureID is what stands in a prompt for each of a picture's tokens: a
// number below zero that is the picture's own, so that the pool of read
// prompts tells one picture from another. The model is given the tokenizer's
// picture token for it.
func inkPictureID(data []byte) int32 {
	h := fnv.New64a()
	h.Write(data)
	return -1 - int32(h.Sum64()&0x3fffffff)
}

// inkPixels decodes a picture and lays it out for the tower: whole tiles of
// tile × tile pixels, rows × cols of them.
func inkPixels(data []byte, tile int) (pixels []float32, rows, cols int, err error) {
	img, err := picture.Decode(data)
	if err != nil {
		return nil, 0, 0, engine.PictureError{Err: err}
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	rows, cols = (h-1)/tile+1, (w-1)/tile+1
	if rows*cols > inkPictureTokens {
		// Made smaller, to whole tiles, its shape kept as nearly as they allow.
		s := math.Sqrt(float64(inkPictureTokens) * float64(tile*tile) / (float64(w) * float64(h)))
		rows, cols = max(1, int(float64(h)*s)/tile), max(1, int(float64(w)*s)/tile)
		rows, cols = min(rows, inkPictureTokens), min(cols, inkPictureTokens)
		if rows*cols > inkPictureTokens {
			if rows > cols {
				rows = inkPictureTokens / cols
			} else {
				cols = inkPictureTokens / rows
			}
		}
		rgb := picture.ResizeRGB(img, cols*tile, rows*tile)
		pixels = make([]float32, len(rgb))
		for i, v := range rgb {
			pixels[i] = mlxmodel.InklingPixel(int(v), i%3)
		}
		return pixels, rows, cols, nil
	}
	// As it is: what is beyond its edge, to the next whole tile, is −1, and
	// what shows through it is a checkerboard, as Ollama lays one out.
	height, width := rows*tile, cols*tile
	pixels = make([]float32, height*width*3)
	pad := [3]float32{mlxmodel.InklingPixel(-1, 0), mlxmodel.InklingPixel(-1, 1), mlxmodel.InklingPixel(-1, 2)}
	opaque := opaquePicture(img)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			o := 3 * (y*width + x)
			if x >= w || y >= h {
				copy(pixels[o:o+3], pad[:])
				continue
			}
			p := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			rgb := [3]int{int(p.R), int(p.G), int(p.B)}
			if !opaque && p.A != 255 {
				bg := 204
				if ((x/8)+(y/8))%2 == 1 {
					bg = 153
				}
				for ch := range rgb {
					rgb[ch] = (rgb[ch]*int(p.A) + bg*(255-int(p.A)) + 127) / 255
				}
			}
			for ch, v := range rgb {
				pixels[o+ch] = mlxmodel.InklingPixel(v, ch)
			}
		}
	}
	return pixels, rows, cols, nil
}

func opaquePicture(img image.Image) bool {
	o, ok := img.(interface{ Opaque() bool })
	return ok && o.Opaque()
}

// pictureSpans is what the pictures of j give a pass over prompt[read:end]:
// the pass's tokens, with the picture token where a picture's are, and the
// features that stand there. Each picture goes through the tower when a pass
// first reaches it. free releases what the pass alone used.
func (e *Inkling) pictureSpans(j *inkJob, read, end int) (toks []int32, spans []mlxmodel.Inkling2Span, free func(), err error) {
	toks = j.prompt[read:end]
	var cuts []mlx.Array
	free = func() {
		for _, a := range cuts {
			a.Free()
		}
	}
	for _, pic := range j.pictures {
		lo, hi := max(read, pic.at), min(end, pic.at+pic.n)
		if lo >= hi {
			continue
		}
		if pic.features == 0 {
			if pic.features, err = e.model.EncodePicture(pic.pixels, pic.rows, pic.cols); err != nil {
				return nil, nil, free, fmt.Errorf("reading a picture: %w", err)
			}
			pic.pixels = nil
		}
		f := pic.features
		if hi-lo < pic.n {
			f = mlx.Slice(f, []int{lo - pic.at, 0}, []int{hi - pic.at, f.Shape()[1]})
			cuts = append(cuts, f)
		}
		spans = append(spans, mlxmodel.Inkling2Span{At: lo - read, Features: f})
	}
	if len(spans) > 0 {
		toks = append([]int32(nil), toks...)
		for i, t := range toks {
			if t < 0 {
				toks[i] = e.pictureTok
			}
		}
	}
	return toks, spans, free, nil
}
