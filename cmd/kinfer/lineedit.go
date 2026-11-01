package main

// The REPL's line editor.
//
// A terminal's own line editing, which bufio.Scanner relied on, erases a byte
// per backspace, not a character: backspacing over Chinese, three bytes a
// character, left two of them in the line and half a character on the screen,
// and the request went out with broken UTF-8. So on a terminal the REPL edits
// its lines itself, in raw mode: by character, two columns for a wide one,
// with the cursor keys, history, and pastes kept whole. Piped, it reads lines.

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode"
)

// errInterrupted is Ctrl-C on an empty line: the REPL ends, as at Ctrl-D.
var errInterrupted = errors.New("interrupted")

type lineReader struct {
	fd      int
	tty     bool
	r       *bufio.Reader
	out     io.Writer
	history []string
}

func newLineReader(in *os.File, out io.Writer) *lineReader {
	fd := int(in.Fd())
	return &lineReader{fd: fd, tty: isTerminal(fd), r: bufio.NewReaderSize(in, 64<<10), out: out}
}

// ReadLine shows prompt and returns the line typed, without its newline: io.EOF
// at the end of input or Ctrl-D on an empty line, errInterrupted at Ctrl-C on
// one.
func (lr *lineReader) ReadLine(prompt string) (string, error) {
	if lr.tty {
		if restore, err := makeRaw(lr.fd); err == nil {
			defer restore()
			return lr.edit(prompt)
		}
	}
	fmt.Fprint(lr.out, prompt)
	line, err := lr.r.ReadString('\n')
	if err != nil && (line == "" || !errors.Is(err, io.EOF)) {
		return "", err
	}
	return strings.TrimRight(line, "\r\n"), nil
}

func (lr *lineReader) edit(prompt string) (string, error) {
	// Bracketed paste: the terminal marks a paste, so its newlines are text,
	// not Enter pressed once a line.
	fmt.Fprint(lr.out, "\x1b[?2004h")
	defer fmt.Fprint(lr.out, "\x1b[?2004l")
	e := &editor{prompt: prompt, history: lr.history, hist: len(lr.history), cols: termWidth(lr.fd)}
	lr.draw(e)
	for {
		k, err := readKey(lr.r)
		if err != nil {
			return "", err
		}
		switch e.handle(k) {
		case editSubmit:
			e.pos = len(e.buf)
			lr.draw(e)
			fmt.Fprint(lr.out, "\r\n")
			line := string(e.buf)
			if strings.TrimSpace(line) != "" && (len(lr.history) == 0 || lr.history[len(lr.history)-1] != line) {
				lr.history = append(lr.history, line)
			}
			return line, nil
		case editEOF:
			fmt.Fprint(lr.out, "\r\n")
			return "", io.EOF
		case editInterrupt:
			fmt.Fprint(lr.out, "\r\n")
			return "", errInterrupted
		default:
			lr.draw(e)
		}
	}
}

func (lr *lineReader) draw(e *editor) { fmt.Fprint(lr.out, e.render()) }

// ─── keys ────────────────────────────────────────────────────────────────────

type keyOp int

const (
	keyRune keyOp = iota
	keyPaste
	keyEnter
	keyBackspace
	keyDelete
	keyLeft
	keyRight
	keyHome
	keyEnd
	keyUp
	keyDown
	keyKillToStart // Ctrl-U
	keyKillToEnd   // Ctrl-K
	keyKillWord    // Ctrl-W
	keyCtrlC
	keyCtrlD
	keyNone
)

type key struct {
	op   keyOp
	r    rune   // keyRune
	text string // keyPaste
}

// readKey reads one key: a character, a control key, an escape sequence, or a
// bracketed paste whole.
func readKey(r *bufio.Reader) (key, error) {
	c, _, err := r.ReadRune()
	if err != nil {
		return key{}, err
	}
	switch c {
	case '\r', '\n':
		return key{op: keyEnter}, nil
	case 127, 8:
		return key{op: keyBackspace}, nil
	case 1:
		return key{op: keyHome}, nil
	case 2:
		return key{op: keyLeft}, nil
	case 3:
		return key{op: keyCtrlC}, nil
	case 4:
		return key{op: keyCtrlD}, nil
	case 5:
		return key{op: keyEnd}, nil
	case 6:
		return key{op: keyRight}, nil
	case 11:
		return key{op: keyKillToEnd}, nil
	case 14:
		return key{op: keyDown}, nil
	case 16:
		return key{op: keyUp}, nil
	case 21:
		return key{op: keyKillToStart}, nil
	case 23:
		return key{op: keyKillWord}, nil
	case 27:
		return readEscape(r)
	}
	if c < 32 || c == unicode.ReplacementChar {
		return key{op: keyNone}, nil
	}
	return key{op: keyRune, r: c}, nil
}

func readEscape(r *bufio.Reader) (key, error) {
	c, _, err := r.ReadRune()
	if err != nil {
		return key{}, err
	}
	if c != '[' && c != 'O' {
		return key{op: keyNone}, nil // Alt with a key
	}
	var params strings.Builder
	for {
		f, _, err := r.ReadRune()
		if err != nil {
			return key{}, err
		}
		if f >= 0x30 && f <= 0x3f { // digits and ;
			params.WriteRune(f)
			continue
		}
		switch f {
		case 'A':
			return key{op: keyUp}, nil
		case 'B':
			return key{op: keyDown}, nil
		case 'C':
			return key{op: keyRight}, nil
		case 'D':
			return key{op: keyLeft}, nil
		case 'H':
			return key{op: keyHome}, nil
		case 'F':
			return key{op: keyEnd}, nil
		case '~':
			switch params.String() {
			case "1", "7":
				return key{op: keyHome}, nil
			case "4", "8":
				return key{op: keyEnd}, nil
			case "3":
				return key{op: keyDelete}, nil
			case "200":
				return readPaste(r)
			}
		}
		return key{op: keyNone}, nil
	}
}

// readPaste reads a bracketed paste up to its end marker, newlines kept.
func readPaste(r *bufio.Reader) (key, error) {
	const end = "\x1b[201~"
	var b strings.Builder
	for {
		c, _, err := r.ReadRune()
		if err != nil {
			return key{}, err
		}
		b.WriteRune(c)
		if s := b.String(); strings.HasSuffix(s, end) {
			s = strings.TrimSuffix(s, end)
			s = strings.ReplaceAll(strings.ReplaceAll(s, "\r\n", "\n"), "\r", "\n")
			return key{op: keyPaste, text: s}, nil
		}
	}
}

// ─── editing ─────────────────────────────────────────────────────────────────

type editResult int

const (
	editContinue editResult = iota
	editSubmit
	editEOF
	editInterrupt
)

type editor struct {
	prompt  string
	buf     []rune
	pos     int
	history []string
	hist    int    // the history entry shown; len(history) is the line being typed
	typed   []rune // that line, while history is shown
	cols    int
	row     int // the cursor's row below the prompt's, as last drawn
}

func (e *editor) insert(rs []rune) {
	e.buf = append(e.buf[:e.pos], append(rs, e.buf[e.pos:]...)...)
	e.pos += len(rs)
}

func (e *editor) handle(k key) editResult {
	switch k.op {
	case keyRune:
		e.insert([]rune{k.r})
	case keyPaste:
		e.insert([]rune(k.text))
	case keyEnter:
		return editSubmit
	case keyBackspace:
		if e.pos > 0 {
			e.buf = append(e.buf[:e.pos-1], e.buf[e.pos:]...)
			e.pos--
		}
	case keyDelete:
		if e.pos < len(e.buf) {
			e.buf = append(e.buf[:e.pos], e.buf[e.pos+1:]...)
		}
	case keyLeft:
		e.pos = max(e.pos-1, 0)
	case keyRight:
		e.pos = min(e.pos+1, len(e.buf))
	case keyHome:
		e.pos = 0
	case keyEnd:
		e.pos = len(e.buf)
	case keyKillToStart:
		e.buf, e.pos = append([]rune{}, e.buf[e.pos:]...), 0
	case keyKillToEnd:
		e.buf = e.buf[:e.pos]
	case keyKillWord:
		i := e.pos
		for i > 0 && unicode.IsSpace(e.buf[i-1]) {
			i--
		}
		for i > 0 && !unicode.IsSpace(e.buf[i-1]) {
			i--
		}
		e.buf, e.pos = append(e.buf[:i], e.buf[e.pos:]...), i
	case keyUp:
		if e.hist > 0 {
			if e.hist == len(e.history) {
				e.typed = append([]rune{}, e.buf...)
			}
			e.hist--
			e.buf = []rune(e.history[e.hist])
			e.pos = len(e.buf)
		}
	case keyDown:
		if e.hist < len(e.history) {
			e.hist++
			if e.hist == len(e.history) {
				e.buf = e.typed
			} else {
				e.buf = []rune(e.history[e.hist])
			}
			e.pos = len(e.buf)
		}
	case keyCtrlC:
		if len(e.buf) == 0 {
			return editInterrupt
		}
		e.buf, e.pos = nil, 0
	case keyCtrlD:
		if len(e.buf) == 0 {
			return editEOF
		}
		if e.pos < len(e.buf) {
			e.buf = append(e.buf[:e.pos], e.buf[e.pos+1:]...)
		}
	}
	return editContinue
}

// ─── drawing ─────────────────────────────────────────────────────────────────

// runeWidth is the columns a character takes: two for East Asian wide and
// full-width characters and most emoji, none for combining marks.
func runeWidth(r rune) int {
	switch {
	case r == 0, unicode.Is(unicode.Mn, r), unicode.Is(unicode.Me, r), unicode.Is(unicode.Cf, r):
		return 0
	case r >= 0x1100 && r <= 0x115f, // Hangul Jamo
		r >= 0x2e80 && r <= 0x303e, // CJK radicals, punctuation
		r >= 0x3041 && r <= 0x33ff, // kana, CJK symbols
		r >= 0x3400 && r <= 0x4dbf, // CJK extension A
		r >= 0x4e00 && r <= 0x9fff, // CJK unified ideographs
		r >= 0xa000 && r <= 0xa4cf, // Yi
		r >= 0xac00 && r <= 0xd7a3, // Hangul syllables
		r >= 0xf900 && r <= 0xfaff, // CJK compatibility ideographs
		r >= 0xfe30 && r <= 0xfe4f, // CJK compatibility forms
		r >= 0xff00 && r <= 0xff60, // full-width forms
		r >= 0xffe0 && r <= 0xffe6,
		r >= 0x1f300 && r <= 0x1f64f, // emoji
		r >= 0x1f900 && r <= 0x1f9ff,
		r >= 0x20000 && r <= 0x3fffd: // CJK extensions B on
		return 2
	}
	return 1
}

// shown is how a character in the line is drawn: a pasted newline as ↵.
func shown(r rune) rune {
	if r == '\n' {
		return '↵'
	}
	return r
}

// place advances a cursor at row, col past a character w columns wide on a
// terminal cols wide: a wide character that does not fit the row starts the
// next, and a full row leaves the cursor at the start of the next.
func place(row, col, w, cols int) (int, int) {
	if w == 0 {
		return row, col
	}
	if col+w > cols {
		row, col = row+1, 0
	}
	col += w
	if col >= cols {
		row, col = row+1, 0
	}
	return row, col
}

// render redraws the prompt and the line from the prompt's row, and leaves
// the cursor at pos. A line longer than the terminal wraps, and the rows are
// counted to come back up for the next redraw.
func (e *editor) render() string {
	var b strings.Builder
	if e.row > 0 {
		fmt.Fprintf(&b, "\x1b[%dA", e.row)
	}
	b.WriteString("\r\x1b[J")
	b.WriteString(e.prompt)
	row, col := 0, 0
	for _, r := range e.prompt {
		row, col = place(row, col, runeWidth(r), e.cols)
	}
	curRow, curCol := row, col
	for i, r := range e.buf {
		r = shown(r)
		b.WriteRune(r)
		row, col = place(row, col, runeWidth(r), e.cols)
		if i+1 == e.pos {
			curRow, curCol = row, col
		}
	}
	// A line that ends exactly at the edge leaves the terminal's cursor on the
	// last column, not the next row's first; a newline puts it where counted.
	if col == 0 && row > 0 && len(e.buf) > 0 {
		b.WriteString("\n")
	}
	if row > curRow {
		fmt.Fprintf(&b, "\x1b[%dA", row-curRow)
	}
	b.WriteString("\r")
	if curCol > 0 {
		fmt.Fprintf(&b, "\x1b[%dC", curCol)
	}
	e.row = curRow
	return b.String()
}
