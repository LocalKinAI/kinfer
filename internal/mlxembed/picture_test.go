package mlxembed

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"image"
	"image/color"
	"image/png"
	"testing"
)

// A PNG whose header says it is far larger than it is is refused from the
// header, before anything is decoded; a small one is decoded as before.
func TestAPictureIsSizedFromItsHeaderFirst(t *testing.T) {
	small := image.NewRGBA(image.Rect(0, 0, 3, 2))
	small.Set(1, 1, color.RGBA{200, 10, 10, 255})
	var b bytes.Buffer
	if err := png.Encode(&b, small); err != nil {
		t.Fatal(err)
	}
	img, err := decodePicture(b.Bytes())
	if err != nil || img.Bounds().Dx() != 3 || img.Bounds().Dy() != 2 {
		t.Fatalf("a 3×2 PNG: %v, %v", img, err)
	}

	// The same file, its IHDR saying 50,000 × 50,000: ten gigabytes decoded.
	bomb := append([]byte(nil), b.Bytes()...)
	binary.BigEndian.PutUint32(bomb[16:], 50000)
	binary.BigEndian.PutUint32(bomb[20:], 50000)
	binary.BigEndian.PutUint32(bomb[29:], crc32.ChecksumIEEE(bomb[12:29]))
	if _, err := decodePicture(bomb); !errors.Is(err, errTooLarge) {
		t.Errorf("a %d-byte PNG saying 50000×50000: %v, want it refused as too large", len(bomb), err)
	}
}
