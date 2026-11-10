package mlxasr

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
)

// Decode turns a recording into 16 kHz mono samples. A WAV at 16 kHz is read
// here; anything else — another rate, mp3, m4a, flac, aiff — goes through
// macOS's afconvert first, which every Mac has, as MLX runs on nothing else.
func Decode(b []byte) ([]float32, error) {
	if pcm, rate, err := readWAV(b); err == nil && rate == sampleRate {
		return pcm, nil
	}
	return convert(b)
}

func convert(b []byte) ([]float32, error) {
	tool, err := exec.LookPath("afconvert")
	if err != nil {
		return nil, fmt.Errorf("the recording is not a 16 kHz WAV, and afconvert, which converts others, is not here: %w", err)
	}
	dir, err := os.MkdirTemp("", "kinfer-asr-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	in, out := filepath.Join(dir, "in"), filepath.Join(dir, "out.wav")
	if err := os.WriteFile(in, b, 0o600); err != nil {
		return nil, err
	}
	if msg, err := exec.Command(tool, "-f", "WAVE", "-d", "LEF32@16000", "-c", "1", in, out).CombinedOutput(); err != nil {
		return nil, fmt.Errorf("not a recording afconvert reads (wav, mp3, m4a, flac, aiff…): %s", msg)
	}
	w, err := os.ReadFile(out)
	if err != nil {
		return nil, err
	}
	pcm, rate, err := readWAV(w)
	if err != nil {
		return nil, err
	}
	if rate != sampleRate {
		return nil, fmt.Errorf("afconvert wrote %d Hz, not %d", rate, sampleRate)
	}
	return pcm, nil
}

// readWAV reads a WAV's samples, mono — the channels averaged — as float32 in
// [-1, 1]: 8, 16, 24 or 32-bit PCM, or 32-bit float.
func readWAV(b []byte) ([]float32, int, error) {
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a WAV")
	}
	var format, channels, bits uint16
	var rate uint32
	for off := 12; off+8 <= len(b); {
		id, size := string(b[off:off+4]), int(binary.LittleEndian.Uint32(b[off+4:off+8]))
		body := off + 8
		if size < 0 || body+size > len(b) || size == 0xffffffff {
			size = len(b) - body // a header written before the length was known
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, 0, fmt.Errorf("a WAV fmt chunk of %d bytes", size)
			}
			format = binary.LittleEndian.Uint16(b[body:])
			channels = binary.LittleEndian.Uint16(b[body+2:])
			rate = binary.LittleEndian.Uint32(b[body+4:])
			bits = binary.LittleEndian.Uint16(b[body+14:])
			if format == 0xfffe && size >= 26 { // WAVE_FORMAT_EXTENSIBLE: the sub-format's first two bytes
				format = binary.LittleEndian.Uint16(b[body+24:])
			}
		case "data":
			if channels == 0 {
				return nil, 0, fmt.Errorf("a WAV with data before its format")
			}
			pcm, err := samples(b[body:body+size], format, int(channels), int(bits))
			return pcm, int(rate), err
		}
		off = body + size + size&1
	}
	return nil, 0, fmt.Errorf("a WAV with no data")
}

func samples(d []byte, format uint16, channels, bits int) ([]float32, error) {
	width := bits / 8
	if width == 0 || channels == 0 {
		return nil, fmt.Errorf("a WAV of %d-bit samples", bits)
	}
	n := len(d) / (width * channels)
	out := make([]float32, n)
	read := func(p []byte) (float32, error) {
		switch {
		case format == 1 && bits == 16:
			return float32(int16(binary.LittleEndian.Uint16(p))) / 32768, nil
		case format == 1 && bits == 24:
			v := int32(uint32(p[0])|uint32(p[1])<<8|uint32(p[2])<<16) << 8 >> 8
			return float32(v) / 8388608, nil
		case format == 1 && bits == 32:
			return float32(int32(binary.LittleEndian.Uint32(p))) / 2147483648, nil
		case format == 1 && bits == 8:
			return (float32(p[0]) - 128) / 128, nil
		case format == 3 && bits == 32:
			return math.Float32frombits(binary.LittleEndian.Uint32(p)), nil
		}
		return 0, fmt.Errorf("a WAV of format %d, %d-bit", format, bits)
	}
	for i := 0; i < n; i++ {
		var sum float32
		for c := 0; c < channels; c++ {
			v, err := read(d[(i*channels+c)*width:])
			if err != nil {
				return nil, err
			}
			sum += v
		}
		out[i] = sum / float32(channels)
	}
	return out, nil
}
