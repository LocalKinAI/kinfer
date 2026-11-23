// Package picture reads the pictures a caller sends: decoding, with what Go
// does not read converted by macOS first, and resizing.
package picture

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	_ "image/gif"  // decoders, registered
	_ "image/jpeg" //
	_ "image/png"  //
	"os"
	"os/exec"
	"path/filepath"
)

// MaxPixels is the most pixels a picture may have: a phone's largest
// photo is 48 million. A picture is decoded whole before it is made small, and
// its header is all that says how large it is: a few hundred kilobytes of PNG
// can say 50,000 × 50,000, gigabytes once decoded.
const MaxPixels = 100_000_000

// Decode decodes a picture, its size read from its header first and
// refused past MaxPixels; one Go does not read (WebP, HEIC, TIFF) is
// made a PNG first.
func Decode(data []byte) (image.Image, error) {
	decode := func(b []byte) (image.Image, error) {
		cfg, _, err := image.DecodeConfig(bytes.NewReader(b))
		if err != nil {
			return nil, err
		}
		if cfg.Width <= 0 || cfg.Height <= 0 || int64(cfg.Width)*int64(cfg.Height) > MaxPixels {
			return nil, fmt.Errorf("%w: %d×%d is more than %d million pixels", ErrTooLarge, cfg.Width, cfg.Height, MaxPixels/1_000_000)
		}
		img, _, err := image.Decode(bytes.NewReader(b))
		return img, err
	}
	img, err := decode(data)
	if errors.Is(err, ErrTooLarge) {
		return nil, err
	}
	if err != nil {
		png, cerr := toPNG(data) // WebP, HEIC, TIFF
		if cerr != nil {
			return nil, fmt.Errorf("not a picture this reads: %w", err)
		}
		if img, err = decode(png); err != nil {
			if errors.Is(err, ErrTooLarge) {
				return nil, err
			}
			return nil, fmt.Errorf("not a picture this reads: %w", err)
		}
	}
	return img, nil
}

var ErrTooLarge = errors.New("the picture is too large")

// ConvertWith runs tool on data in a scratch folder: args with "IN" and
// "OUT" standing for the two files; it returns what was written to OUT.
func ConvertWith(tool string, data []byte, args ...string) ([]byte, error) {
	path, err := exec.LookPath(tool)
	if err != nil {
		return nil, fmt.Errorf("%s, which converts it, is not here: %w", tool, err)
	}
	dir, err := os.MkdirTemp("", "kinfer-convert-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	in, out := filepath.Join(dir, "in"), filepath.Join(dir, "out")
	if err := os.WriteFile(in, data, 0o600); err != nil {
		return nil, err
	}
	for i, a := range args {
		switch a {
		case "IN":
			args[i] = in
		case "OUT":
			args[i] = out
		}
	}
	if msg, err := exec.Command(path, args...).CombinedOutput(); err != nil {
		return nil, fmt.Errorf("%s could not read it: %s", tool, msg)
	}
	return os.ReadFile(out)
}

// toPNG is a picture Go cannot decode, as a PNG.
func toPNG(data []byte) ([]byte, error) {
	return ConvertWith("sips", data, "-s", "format", "png", "IN", "--out", "OUT")
}
