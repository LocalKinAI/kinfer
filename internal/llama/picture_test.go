package llama

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/png"
	"strings"
	"testing"
)

// A picture for a chat model is sized from its header before llama.cpp
// decodes it whole: a PNG of a few dozen bytes saying 50000×50000, or a BMP
// saying as much, is refused; a real picture goes through.
func TestAPictureTooLargeIsRefusedByItsHeader(t *testing.T) {
	var small bytes.Buffer
	png.Encode(&small, image.NewRGBA(image.Rect(0, 0, 64, 48)))
	if err := checkPicture(0, small.Bytes()); err != nil {
		t.Fatalf("a 64×48 PNG: %v", err)
	}
	bomb := append([]byte(nil), small.Bytes()[:33]...)
	binary.BigEndian.PutUint32(bomb[16:], 50000)
	binary.BigEndian.PutUint32(bomb[20:], 50000)
	binary.BigEndian.PutUint32(bomb[29:], crc32.ChecksumIEEE(bomb[12:29]))
	if err := checkPicture(1, bomb); err == nil || !strings.Contains(err.Error(), "50000×50000") {
		t.Errorf("a %d-byte PNG saying 50000×50000: %v", len(bomb), err)
	}
	bmp := make([]byte, 54)
	copy(bmp, "BM")
	binary.LittleEndian.PutUint32(bmp[18:], 40000)
	binary.LittleEndian.PutUint32(bmp[22:], uint32(0xFFFFFFFF-39999)) // -40000: stored top down
	if err := checkPicture(2, bmp); err == nil {
		t.Error("a BMP saying 40000×40000 was let through")
	}
	if err := checkPicture(3, []byte("RIFF....WAVEfmt ")); err != nil {
		t.Errorf("a sound is not a picture to size: %v", err)
	}
}
