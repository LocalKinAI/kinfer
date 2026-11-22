package mlxasr

import (
	"context"
	"strings"
	"unicode"
	"unicode/utf8"
)

// A long recording is read a piece at a time. Qwen3-ASR reads twenty minutes
// whole, and twenty-five; given an hour in one pass it wrote down the first
// seventeen and a half minutes and ended its reply as if that were all — no
// error, no bound reached, 160 sentences of 550. mlx-audio, which this is a
// port of, cuts at twenty minutes, where the recording is quietest within
// five seconds either side, and so does this.
const (
	pieceSeconds     = 1200
	cutSearchSeconds = 5
	cutWindowSamples = sampleRate / 10 // the quiet is looked for a tenth of a second at a time
)

// pieces cuts pcm into pieces of at most about pieceSeconds, each cut at the
// quietest tenth of a second within cutSearchSeconds of where it falls.
func pieces(pcm []float32) [][]float32 {
	const most, search = pieceSeconds * sampleRate, cutSearchSeconds * sampleRate
	var out [][]float32
	for start := 0; ; {
		end := start + most
		if end >= len(pcm) {
			return append(out, pcm[start:])
		}
		lo, hi := max(start, end-search), min(len(pcm), end+search)
		cut := end
		if hi-lo > cutWindowSamples {
			var sum float64
			for _, v := range pcm[lo : lo+cutWindowSamples] {
				sum += float64(v) * float64(v)
			}
			least, at := sum, lo
			for i := lo + 1; i+cutWindowSamples <= hi; i++ {
				a, b := float64(pcm[i-1]), float64(pcm[i+cutWindowSamples-1])
				if sum += b*b - a*a; sum < least {
					least, at = sum, i
				}
			}
			cut = at + cutWindowSamples/2
		}
		cut = max(cut, start+sampleRate) // a piece is a second at least
		out = append(out, pcm[start:cut])
		start = cut
	}
}

// opener is what Qwen3-ASR starts a reply with when it was not told the
// language: "language English<asr_text>". after is the reply past it.
const openerEnd = "<asr_text>"

func afterOpener(reply string) string {
	if i := strings.Index(reply, openerEnd); i >= 0 {
		return reply[i+len(openerEnd):]
	}
	return reply
}

// joint is what goes between two pieces' transcripts: a space, or nothing
// after a script written without them.
func joint(before string) string {
	r, _ := utf8.DecodeLastRuneInString(strings.TrimRight(before, " \n"))
	if before == "" || unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r) ||
		(r >= 0x3000 && r <= 0x303F) || (r >= 0xFF00 && r <= 0xFFEF) {
		return ""
	}
	return " "
}

// Transcribe writes down what pcm, 16 kHz mono, says. language, when not "",
// starts the reply as Qwen3-ASR is told one; onToken, when not nil, gets the
// text as it is written.
//
// A recording longer than twenty minutes is read in pieces (see pieces), and
// the result is the pieces' transcripts joined: the first piece's reply as
// the model wrote it, the language it heard at its head, and each later
// piece's text after it. Each piece has its own bound, maxTokens or its own
// length in sound, and hears its language for itself.
func (m *Model) Transcribe(ctx context.Context, pcm []float32, system, language string, maxTokens int, onToken func(string)) (Result, error) {
	parts := pieces(pcm)
	if len(parts) == 1 {
		return m.transcribe(ctx, pcm, system, language, maxTokens, onToken)
	}
	var all Result
	for i, part := range parts {
		cb, flush := onToken, func() {}
		if i > 0 && onToken != nil {
			// A later piece's text goes on from the last one's; the language
			// it opens with, when it was not told one, is not sent again.
			sep, held, open := joint(all.Text), "", language == ""
			cb = func(s string) {
				if !open {
					onToken(sep + s)
					sep = ""
					return
				}
				held += s
				if j := strings.Index(held, openerEnd); j >= 0 {
					open = false
					if rest := held[j+len(openerEnd):]; rest != "" {
						onToken(sep + rest)
						sep = ""
					}
				}
			}
			flush = func() { // a reply that never opened as expected: all of it is text
				if open && held != "" {
					onToken(sep + held)
				}
			}
		}
		res, err := m.transcribe(ctx, part, system, language, maxTokens, cb)
		flush()
		all.PromptTokens += res.PromptTokens
		all.AudioTokens += res.AudioTokens
		all.Tokens += res.Tokens
		all.Encode += res.Encode
		all.Prefill += res.Prefill
		all.Decode += res.Decode
		all.Truncated = all.Truncated || res.Truncated
		all.ForcedLanguage = res.ForcedLanguage
		if i == 0 {
			all.Text = res.Text
		} else {
			text := res.Text
			if language == "" {
				text = afterOpener(text)
			}
			all.Text += joint(all.Text) + text
		}
		if err != nil {
			return all, err
		}
	}
	return all, nil
}
