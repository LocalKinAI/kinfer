package mlxembed

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// What Go and Ollama do not read, macOS does: a WebP, HEIC or TIFF picture
// through sips, and an mp3, m4a, aac or ogg through afconvert — the tools
// every Mac has, as MLX runs on nothing else (internal/mlxasr converts sound
// the same way). Ollama refuses these; the home keeps a few of each.

// convertWith runs tool on data in a scratch folder: args with "IN" and
// "OUT" standing for the two files; it returns what was written to OUT.
func convertWith(tool string, data []byte, args ...string) ([]byte, error) {
	path, err := exec.LookPath(tool)
	if err != nil {
		return nil, fmt.Errorf("%s, which converts it, is not here: %w", tool, err)
	}
	dir, err := os.MkdirTemp("", "kinfer-embed-")
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
	return convertWith("sips", data, "-s", "format", "png", "IN", "--out", "OUT")
}

// toWAV is a sound that is not a WAV, as a float WAV at its own rate, its
// channels mixed; resample then makes it 16 kHz as for any WAV.
func toWAV(data []byte) ([]byte, error) {
	return convertWith("afconvert", data, "-f", "WAVE", "-d", "LEF32", "-c", "1", "IN", "OUT")
}
