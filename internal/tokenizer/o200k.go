package tokenizer

import "unicode"

// scanO200k splits s as o200kPattern does, its alternatives tried in order
// at each position and each as greedy as the regex is, giving back what a
// later part needs.
func scanO200k(s string) []string {
	var out []string
	r := []rune(s)
	for i := 0; i < len(r); {
		j := matchO200k(r, i)
		if j <= i {
			j = i + 1
		}
		out = append(out, string(r[i:j]))
		i = j
	}
	return out
}

// The pattern's two classes of letter: what may open a word — upper and title
// case — and what its body is made of, lower case; letters with no case,
// modifiers and marks are both.
func o200kUpper(c rune) bool {
	return unicode.In(c, unicode.Lu, unicode.Lt, unicode.Lm, unicode.Lo, unicode.M)
}

func o200kLower(c rune) bool {
	return unicode.In(c, unicode.Ll, unicode.Lm, unicode.Lo, unicode.M)
}

// o200kSuffix is the end of an optional 's, 't, 're, 've, 'm, 'll or 'd at i,
// in either case.
func o200kSuffix(r []rune, i int) int {
	n := len(r)
	if i+1 >= n || r[i] != '\'' {
		return i
	}
	a := unicode.ToLower(r[i+1])
	// 's, 't, 'm and 'd come before the longer ones only where the regex
	// lists them so: 's|'t|'re|'ve|'m|'ll|'d.
	if a == 's' || a == 't' {
		return i + 2
	}
	if i+2 < n {
		b := unicode.ToLower(r[i+2])
		if (a == 'r' && b == 'e') || (a == 'v' && b == 'e') {
			return i + 3
		}
	}
	if a == 'm' {
		return i + 2
	}
	if i+2 < n && a == 'l' && unicode.ToLower(r[i+2]) == 'l' {
		return i + 3
	}
	if a == 'd' {
		return i + 2
	}
	return i
}

// o200kWord is the end of [upper]*[lower]+ at s, or -1: as many openers as
// leave a body after them.
func o200kWord(r []rune, s int) int {
	n := len(r)
	a := s
	for a < n && o200kUpper(r[a]) {
		a++
	}
	b := a
	for b < n && o200kLower(r[b]) {
		b++
	}
	if b > a {
		return b
	}
	// Nothing lower follows the openers: the body is the last of them that
	// can be one, and the word ends with it.
	for p := a - 1; p >= s; p-- {
		if o200kLower(r[p]) {
			return p + 1
		}
	}
	return -1
}

// o200kCaps is the end of [upper]+[lower]* at s, or -1.
func o200kCaps(r []rune, s int) int {
	n := len(r)
	a := s
	for a < n && o200kUpper(r[a]) {
		a++
	}
	if a == s {
		return -1
	}
	for a < n && o200kLower(r[a]) {
		a++
	}
	return a
}

func matchO200k(r []rune, i int) int {
	n := len(r)
	c := r[i]
	lead := !isNL(c) && !unicode.IsLetter(c) && !isN(c) // [^\r\n\p{L}\p{N}]

	// The two kinds of word, each with the character before it first and
	// then without.
	for _, word := range []func([]rune, int) int{o200kWord, o200kCaps} {
		if lead && i+1 < n {
			if e := word(r, i+1); e > 0 {
				return o200kSuffix(r, e)
			}
		}
		if e := word(r, i); e > 0 {
			return o200kSuffix(r, e)
		}
	}

	// \p{N}{1,3}
	if isN(c) {
		k := i + 1
		for k < n && k < i+3 && isN(r[k]) {
			k++
		}
		return k
	}

	// ' ?[^\s\p{L}\p{N}]+[\r\n/]*'
	other := func(x rune) bool { return !isSpace(x) && !unicode.IsLetter(x) && !isN(x) }
	k := i
	if c == ' ' && i+1 < n && other(r[i+1]) {
		k = i + 1
	}
	if other(r[k]) {
		for k < n && other(r[k]) {
			k++
		}
		for k < n && (isNL(r[k]) || r[k] == '/') {
			k++
		}
		return k
	}

	// The rest start with whitespace.
	end := i
	for end < n && isSpace(r[end]) {
		end++
	}
	if end == i {
		return i + 1
	}
	// \s*[\r\n]+ : up to the last newline in the run.
	for k := end - 1; k >= i; k-- {
		if isNL(r[k]) {
			return k + 1
		}
	}
	// \s+(?!\S) : the run, less its last space when text follows.
	if end == n {
		return end
	}
	if end-1 > i {
		return end - 1
	}
	// \s+
	return end
}
