// Package grade judges an answer to a MATH-style problem against the
// reference: Hendrycks' string normalisation of the two LaTeX answers, a few
// unwrappings it lacks, and a numeric comparison when both are numbers.
package grade

import (
	"math"
	"regexp"
	"strconv"
	"strings"
)

// Boxed is what the last \boxed{...} in text holds, braces balanced.
func Boxed(text string) (string, bool) {
	i := strings.LastIndex(text, `\boxed`)
	if i < 0 {
		return "", false
	}
	j := strings.IndexByte(text[i:], '{')
	if j < 0 {
		return "", false
	}
	j += i
	depth := 0
	for k := j; k < len(text); k++ {
		switch text[k] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return text[j+1 : k], true
			}
		}
	}
	return "", false
}

// Same reports whether got and want are the same answer.
func Same(got, want string) bool {
	a, b := Strip(got), Strip(want)
	if a == b {
		return true
	}
	x, okx := number(a)
	y, oky := number(b)
	return okx && oky && math.Abs(x-y) <= 1e-6*math.Max(1, math.Abs(y))
}

var (
	spacing   = strings.NewReplacer(`\,`, "", `\;`, "", `\:`, "", `\ `, "")
	assign    = regexp.MustCompile(`^[a-zA-Z]\w*\s*=`)
	sqrtArg   = regexp.MustCompile(`\\sqrt(\w)`)
	slash     = regexp.MustCompile(`^(-?\d+)/(\d+)$`)
	thousands = regexp.MustCompile(`(\d),(\d\d\d)(\D|$)`)
	plainNum  = regexp.MustCompile(`^-?\d+(\.\d+)?$`)
	fracNum   = regexp.MustCompile(`^(-?)\\frac\{(-?\d+)\}\{(\d+)\}$`)
)

// Strip puts an answer in the form two equal answers share.
func Strip(s string) string {
	s = strings.TrimSpace(s)
	s = strings.NewReplacer("\n", "", `\!`, "").Replace(s)
	s = strings.ReplaceAll(s, `\\`, `\`)
	s = strings.NewReplacer("tfrac", "frac", "dfrac", "frac", `\left`, "", `\right`, "").Replace(s)
	s = strings.NewReplacer(`^{\circ}`, "", `^\circ`, "", "°", "", `\$`, "", "$", "", `\%`, "", "%", "").Replace(s)
	for _, cmd := range []string{"text", "textbf", "mathrm", "mbox", "boxed"} {
		s = unwrap(s, cmd)
	}
	s = spacing.Replace(s)
	s = strings.NewReplacer(" .", " 0.", "{.", "{0.").Replace(s)
	if strings.HasPrefix(s, ".") {
		s = "0" + s
	}
	// "x = 5" is 5; a longer left side is part of the answer.
	if assign.MatchString(s) {
		if lhs, rhs, _ := strings.Cut(s, "="); len(lhs) <= 2 {
			s = rhs
		}
	}
	s = sqrtArg.ReplaceAllString(s, `\sqrt{$1}`)
	s = strings.ReplaceAll(s, " ", "")
	s = fixFracs(s)
	if s == "0.5" {
		s = `\frac{1}{2}`
	}
	s = slash.ReplaceAllString(s, `\frac{$1}{$2}`)
	for {
		t := thousands.ReplaceAllString(s, "$1$2$3")
		if t == s {
			break
		}
		s = t
	}
	return strings.TrimRight(s, ".")
}

// unwrap replaces \cmd{x} with x, braces balanced, everywhere.
func unwrap(s, cmd string) string {
	open := `\` + cmd + "{"
	for {
		i := strings.Index(s, open)
		if i < 0 {
			return s
		}
		j := i + len(open)
		depth, k := 1, j
		for ; k < len(s) && depth > 0; k++ {
			switch s[k] {
			case '{':
				depth++
			case '}':
				depth--
			}
		}
		if depth > 0 {
			return s[:i] + s[j:]
		}
		s = s[:i] + s[j:k-1] + s[k:]
	}
}

// fixFracs braces \frac's arguments: \frac12 and \frac1{2} are \frac{1}{2}.
func fixFracs(s string) string {
	parts := strings.Split(s, `\frac`)
	var b strings.Builder
	b.WriteString(parts[0])
	for _, p := range parts[1:] {
		b.WriteString(`\frac`)
		if strings.HasPrefix(p, "{") || len(p) < 2 {
			b.WriteString(p)
			continue
		}
		if p[1] == '{' {
			b.WriteString("{" + p[:1] + "}" + p[1:])
		} else {
			b.WriteString("{" + p[:1] + "}{" + p[1:2] + "}" + p[2:])
		}
	}
	return b.String()
}

func number(s string) (float64, bool) {
	if plainNum.MatchString(s) {
		v, err := strconv.ParseFloat(s, 64)
		return v, err == nil
	}
	if m := fracNum.FindStringSubmatch(s); m != nil {
		num, _ := strconv.Atoi(m[2])
		den, _ := strconv.Atoi(m[3])
		if den == 0 {
			return 0, false
		}
		v := float64(num) / float64(den)
		if m[1] == "-" {
			v = -v
		}
		return v, true
	}
	return 0, false
}
