package mlxasr

import (
	"fmt"
	"math"
	"sync"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// Whisper's log-mel features, as transformers' WhisperFeatureExtractor makes
// them for Qwen3-ASR (preprocessor_config.json): 16 kHz, a 400-sample periodic
// Hann window every 160 samples over the recording padded by reflection at both
// ends, the power spectrum through 128 Slaney mel filters up to 8 kHz, log10
// floored at 1e-10, the last frame dropped, floored again at the largest value
// less 8, and (x + 4) / 4. The frames are cut here; the FFT and what follows
// run on the GPU.
const (
	sampleRate = 16000
	nFFT       = 400
	hop        = 160
	nFreq      = nFFT/2 + 1
)

var (
	melOnce   sync.Once
	melBank   []float32 // [nFreq, mels], row-major
	melBankN  int
	hannTable []float32
)

func slaneyHzToMel(f float64) float64 {
	const minLogHz, minLogMel = 1000.0, 15.0
	logstep := 27.0 / math.Log(6.4)
	if f >= minLogHz {
		return minLogMel + math.Log(f/minLogHz)*logstep
	}
	return 3.0 * f / 200.0
}

func slaneyMelToHz(m float64) float64 {
	const minLogHz, minLogMel = 1000.0, 15.0
	logstep := math.Log(6.4) / 27.0
	if m >= minLogMel {
		return minLogHz * math.Exp(logstep*(m-minLogMel))
	}
	return 200.0 * m / 3.0
}

// melFilters is transformers' mel_filter_bank(1 + n_fft/2, mels, 0, 8000,
// 16000, norm="slaney", mel_scale="slaney"), [nFreq, mels].
func melFilters(mels int) []float32 {
	fft := make([]float64, nFreq)
	for i := range fft {
		fft[i] = float64(i) * (sampleRate / 2.0) / float64(nFreq-1)
	}
	lo, hi := slaneyHzToMel(0), slaneyHzToMel(8000)
	f := make([]float64, mels+2)
	for j := range f {
		f[j] = slaneyMelToHz(lo + (hi-lo)*float64(j)/float64(mels+1))
	}
	out := make([]float32, nFreq*mels)
	for i := 0; i < nFreq; i++ {
		for j := 0; j < mels; j++ {
			down := (fft[i] - f[j]) / (f[j+1] - f[j])
			up := (f[j+2] - fft[i]) / (f[j+2] - f[j+1])
			v := math.Max(0, math.Min(down, up))
			v *= 2.0 / (f[j+2] - f[j]) // Slaney's area normalization
			out[i*mels+j] = float32(v)
		}
	}
	return out
}

func tables(mels int) {
	melOnce.Do(func() {
		melBank, melBankN = melFilters(mels), mels
		hannTable = make([]float32, nFFT)
		for i := range hannTable {
			hannTable[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/nFFT))
		}
	})
}

// logMel is the features of pcm, 16 kHz mono: [mels, frames] float32, one
// frame every 10 ms.
func logMel(pcm []float32, mels int) (mlx.Array, int, error) {
	tables(mels)
	if melBankN != mels {
		return 0, 0, fmt.Errorf("mel bins %d, the filters were made for %d", mels, melBankN)
	}
	n := len(pcm)
	const pad = nFFT / 2
	if n <= pad {
		return 0, 0, fmt.Errorf("the recording is too short: %d samples", n)
	}
	at := func(i int) float32 { // the recording padded by reflection
		switch {
		case i < pad:
			return pcm[pad-i]
		case i >= n+pad:
			return pcm[n-2-(i-n-pad)]
		default:
			return pcm[i-pad]
		}
	}
	T := n / hop // 1 + n/hop frames, less the last
	frames := make([]float32, T*nFFT)
	for t := 0; t < T; t++ {
		row := frames[t*nFFT : (t+1)*nFFT]
		for i := range row {
			row[i] = at(t*hop+i) * hannTable[i]
		}
	}
	x := mlx.FromFloat32(frames, T, nFFT)
	bank := mlx.FromFloat32(melBank, nFreq, mels)
	spec := mlx.Power2(mlx.RFFT(x, nFFT, -1)) // [T, nFreq]
	mel := mlx.Matmul(spec, bank)             // [T, mels]
	mel = mlx.Log10(mlx.Maximum(mel, mlx.Scalar(1e-10)))
	top := mlx.Max(mlx.Reshape(mel, T*mels), 0, false)
	mel = mlx.Maximum(mel, mlx.AddScalar(top, -8))
	mel = mlx.MulScalar(mlx.AddScalar(mel, 4), 0.25)
	return mlx.Transpose(mel, 1, 0), T, nil
}
