package mlxtts

import "math"

// Stretch time-stretches speech to speed (2 is twice as fast) with its pitch
// unchanged, as the Python service's apply_speed does: Qwen3-TTS takes no
// speed of its own. WSOLA — overlapping 30 ms windows, each placed where it
// best continues the last, within ±7.5 ms — which suits speech better than a
// phase vocoder. speed is held to [0.25, 4]; about 1 is the audio as it is.
func Stretch(pcm []float32, speed float64, rate int) []float32 {
	if speed == 0 || math.Abs(speed-1) < 1e-3 || len(pcm) == 0 {
		return pcm
	}
	speed = min(max(speed, 0.25), 4)
	frame := max(64, int(float64(rate)*0.03))
	hopOut := frame / 2
	hopIn := float64(hopOut) * speed
	tol := frame / 4
	window := make([]float32, frame)
	for n := range window { // numpy's hanning: symmetric
		window[n] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/float64(frame-1)))
	}
	outLen := int(float64(len(pcm)) / speed)
	pad := frame + tol
	x := make([]float32, pad+len(pcm)+2*pad)
	copy(x[pad:], pcm)
	y := make([]float32, outLen+2*frame)
	norm := make([]float32, len(y))

	prev := -1
	for k := 0; ; k++ {
		outPos := k * hopOut
		centre := int(math.RoundToEven(float64(k)*hopIn)) + pad
		if outPos+frame > len(y) || centre+tol+frame > len(x) {
			break
		}
		pos := centre
		if prev >= 0 {
			// The frame near centre that best continues the last one.
			natural := x[prev+hopOut : prev+hopOut+frame]
			best, bestAt := math.Inf(-1), 0
			for lag := 0; lag <= 2*tol; lag++ {
				var c float64
				region := x[centre-tol+lag : centre-tol+lag+frame]
				for n := range natural {
					c += float64(region[n]) * float64(natural[n])
				}
				if c > best {
					best, bestAt = c, lag
				}
			}
			pos = centre - tol + bestAt
		}
		for n := 0; n < frame; n++ {
			y[outPos+n] += x[pos+n] * window[n]
			norm[outPos+n] += window[n]
		}
		prev = pos
	}
	start := int(math.RoundToEven(float64(pad) / speed))
	out := make([]float32, outLen)
	for i := range out {
		if j := start + i; j < len(y) {
			out[i] = y[j] / max(norm[j], 1e-3)
		}
	}
	return out
}
