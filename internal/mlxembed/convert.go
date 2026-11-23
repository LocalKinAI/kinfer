package mlxembed

import "github.com/LocalKinAI/kinfer/internal/picture"

// What Go and Ollama do not read, macOS does: a WebP, HEIC or TIFF picture
// through sips, and an mp3, m4a, aac or ogg through afconvert — the tools
// every Mac has, as MLX runs on nothing else (internal/mlxasr converts sound
// the same way). Ollama refuses these; the home keeps a few of each.

// toWAV is a sound that is not a WAV, as a float WAV at its own rate, its
// channels mixed; resample then makes it 16 kHz as for any WAV.
func toWAV(data []byte) ([]byte, error) {
	return picture.ConvertWith("afconvert", data, "-f", "WAVE", "-d", "LEF32", "-c", "1", "IN", "OUT")
}
