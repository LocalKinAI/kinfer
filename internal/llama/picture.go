package llama

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	_ "image/gif"  // DecodeConfig reads their headers
	_ "image/jpeg" //
	_ "image/png"  //
)

// MaxPicturePixels is the most pixels a picture sent to a chat model may have.
// llama.cpp decodes it whole, from what its header says, before anything looks
// at its size: a few hundred bytes of PNG can ask for gigabytes. The embedding
// models' pictures have had the same limit since v0.8.0 (mlxembed).
const MaxPicturePixels = 100_000_000

// pictureSize reads a picture's width and height from its header, for the
// kinds llama.cpp decodes (jpg, png, gif, bmp); ok is false for anything else
// — a sound, or a kind it does not read, which it then refuses itself.
func pictureSize(b []byte) (w, h int, ok bool) {
	if cfg, _, err := image.DecodeConfig(bytes.NewReader(b)); err == nil {
		return cfg.Width, cfg.Height, true
	}
	// BMP: "BM", then the width and height as int32 at 18 and 22 (the height
	// negative for a picture stored top down).
	if len(b) >= 26 && b[0] == 'B' && b[1] == 'M' {
		w := int(int32(binary.LittleEndian.Uint32(b[18:22])))
		h := int(int32(binary.LittleEndian.Uint32(b[22:26])))
		if h < 0 {
			h = -h
		}
		return w, h, true
	}
	return 0, 0, false
}

// checkPicture refuses a picture larger than MaxPicturePixels, by its header.
func checkPicture(i int, b []byte) error {
	w, h, ok := pictureSize(b)
	if !ok {
		return nil
	}
	if w <= 0 || h <= 0 || int64(w)*int64(h) > MaxPicturePixels {
		return fmt.Errorf("picture %d is %d×%d: more than %d million pixels, or not a size", i, w, h, MaxPicturePixels/1_000_000)
	}
	return nil
}
