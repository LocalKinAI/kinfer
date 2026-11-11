package mlxembed

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	_ "image/gif"  // decoders, registered
	_ "image/jpeg" //
	_ "image/png"  //
	"math"
)

// prepare decodes a picture and makes the tower's input at the release's
// budget of soft tokens (vision_soft_tokens_per_image, 280).
func (vt *visionTower) prepare(data []byte, budget int) (visionInput, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		png, cerr := toPNG(data) // WebP, HEIC, TIFF
		if cerr != nil {
			return visionInput{}, fmt.Errorf("not a picture this reads: %w", err)
		}
		if img, _, err = image.Decode(bytes.NewReader(png)); err != nil {
			return visionInput{}, fmt.Errorf("not a picture this reads: %w", err)
		}
	}
	c := vt.cfg
	b := img.Bounds()
	th, tw, err := visionTargetSize(b.Dy(), b.Dx(), c.Patch, c.Pool, budget)
	if err != nil {
		return visionInput{}, err
	}
	if max(th, tw)/c.Patch > vt.positions {
		return visionInput{}, fmt.Errorf("the picture's grid is larger than the tower's %d positions", vt.positions)
	}
	rgb := resizeRGB(img, tw, th)
	return patchify(rgb, tw, th, c.Patch, c.Pool), nil
}

// visionTargetSize is the reference's resize: each side a multiple of
// patch·pool, the area within budget soft tokens, the shape kept; a side
// that floors to nothing is held at one multiple.
func visionTargetSize(height, width, patch, pool, budget int) (int, int, error) {
	if height <= 0 || width <= 0 {
		return 0, 0, fmt.Errorf("a picture of %dx%d", width, height)
	}
	side := float64(patch * pool)
	target := float64(budget) * side * side
	f := math.Sqrt(target / (float64(height) * float64(width)))
	h := math.Floor(f*float64(height)/side) * side
	w := math.Floor(f*float64(width)/side) * side
	if h == 0 && w == 0 {
		return 0, 0, fmt.Errorf("a picture of %dx%d is too small", width, height)
	}
	most := float64(budget) * side
	if h == 0 {
		h = side
		w = min(math.Floor(float64(width)/float64(height))*side, most)
	} else if w == 0 {
		w = side
		h = min(math.Floor(float64(height)/float64(width))*side, most)
	}
	if h*w > target || h > most || w > most {
		return 0, 0, errors.New("the picture does not fit the budget once resized")
	}
	return int(h), int(w), nil
}

// patchify lays the resized picture out as the tower reads it: a row per
// patch, (pixel row, pixel column, RGB) within it, in [0, 1].
func patchify(rgb []uint8, w, h, patch, pool int) visionInput {
	pw, ph := w/patch, h/patch
	n := pw * ph
	per := patch * patch * 3
	out := visionInput{pixels: make([]float32, n*per), positions: make([]int32, 2*n),
		patchesW: pw, patchesH: ph, softTokens: n / (pool * pool), pixelsPerPatch: per}
	for p := 0; p < n; p++ {
		gx, gy := p%pw, p/pw
		out.positions[2*p], out.positions[2*p+1] = int32(gx), int32(gy)
		dst := out.pixels[p*per:]
		for py := 0; py < patch; py++ {
			row := ((gy*patch+py)*w + gx*patch) * 3
			for px := 0; px < patch; px++ {
				o := (py*patch + px) * 3
				dst[o] = float32(rgb[row+px*3]) / 255
				dst[o+1] = float32(rgb[row+px*3+1]) / 255
				dst[o+2] = float32(rgb[row+px*3+2]) / 255
			}
		}
	}
	return out
}
