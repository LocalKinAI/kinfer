package main

import (
	"bufio"
	"strings"
	"testing"
)

// keys reads every key in s, as a terminal would send them.
func keys(t *testing.T, s string) []key {
	r := bufio.NewReader(strings.NewReader(s))
	var ks []key
	for {
		k, err := readKey(r)
		if err != nil {
			return ks
		}
		ks = append(ks, k)
	}
}

func typed(t *testing.T, e *editor, s string) editResult {
	res := editContinue
	for _, k := range keys(t, s) {
		if res = e.handle(k); res != editContinue {
			return res
		}
	}
	return res
}

// The bug this editor exists for: a backspace erases a character, not a byte.
func TestBackspaceErasesAChineseCharacter(t *testing.T) {
	e := &editor{cols: 80}
	if res := typed(t, e, "你好世界\x7f\x7f吗\r"); res != editSubmit {
		t.Fatalf("result %v", res)
	}
	if got := string(e.buf); got != "你好吗" {
		t.Errorf("line %q, want 你好吗", got)
	}
}

func TestCursorKeysAndKills(t *testing.T) {
	e := &editor{cols: 80}
	typed(t, e, "hello world")
	typed(t, e, "\x1b[D\x1b[D\x1b[D\x1b[D\x1b[D") // left x5, before "world"
	typed(t, e, "big ")
	if got := string(e.buf); got != "hello big world" {
		t.Errorf("insert in the middle: %q", got)
	}
	typed(t, e, "\x17") // Ctrl-W: the word before the cursor
	if got := string(e.buf); got != "hello world" {
		t.Errorf("Ctrl-W: %q", got)
	}
	typed(t, e, "\x01\x1b[3~") // Home, Delete
	if got := string(e.buf); got != "ello world" {
		t.Errorf("Home, Delete: %q", got)
	}
	typed(t, e, "\x05\x15") // End, Ctrl-U
	if len(e.buf) != 0 {
		t.Errorf("Ctrl-U at the end left %q", string(e.buf))
	}
}

func TestHistoryKeepsTheLineBeingTyped(t *testing.T) {
	e := &editor{cols: 80, history: []string{"first", "second"}, hist: 2}
	typed(t, e, "draft")
	typed(t, e, "\x1b[A")
	if got := string(e.buf); got != "second" {
		t.Errorf("up: %q", got)
	}
	typed(t, e, "\x1b[A\x1b[A") // past the oldest stays on it
	if got := string(e.buf); got != "first" {
		t.Errorf("up x3: %q", got)
	}
	typed(t, e, "\x1b[B\x1b[B")
	if got := string(e.buf); got != "draft" {
		t.Errorf("back down: %q", got)
	}
}

func TestPasteIsOneLine(t *testing.T) {
	e := &editor{cols: 80}
	if res := typed(t, e, "\x1b[200~line one\r\nline two\x1b[201~"); res != editContinue {
		t.Fatalf("a paste submitted the line: %v", res)
	}
	if got := string(e.buf); got != "line one\nline two" {
		t.Errorf("paste: %q", got)
	}
}

func TestCtrlCAndCtrlD(t *testing.T) {
	e := &editor{cols: 80}
	typed(t, e, "abc")
	if res := typed(t, e, "\x03"); res != editContinue || len(e.buf) != 0 {
		t.Errorf("Ctrl-C with text: %v, %q", res, string(e.buf))
	}
	if res := typed(t, e, "\x03"); res != editInterrupt {
		t.Errorf("Ctrl-C on an empty line: %v", res)
	}
	if res := typed(t, e, "\x04"); res != editEOF {
		t.Errorf("Ctrl-D on an empty line: %v", res)
	}
}

func TestRuneWidth(t *testing.T) {
	for _, c := range []struct {
		r rune
		w int
	}{{'a', 1}, {'你', 2}, {'，', 2}, {'ア', 2}, {'한', 2}, {'é', 1}, {'́', 0}, {'😀', 2}} {
		if got := runeWidth(c.r); got != c.w {
			t.Errorf("runeWidth(%q) = %d, want %d", c.r, got, c.w)
		}
	}
}

// A wide character that does not fit the row starts the next one, and the
// cursor is put back where the counting says.
func TestRenderWrapsWideCharacters(t *testing.T) {
	e := &editor{prompt: "> ", cols: 10}
	e.buf = []rune("你好你好你") // "> " is 2 columns, then 4 characters fill the row; the 5th wraps
	e.pos = len(e.buf)
	out := e.render()
	if e.row != 1 {
		t.Errorf("cursor row %d, want 1", e.row)
	}
	if !strings.HasSuffix(out, "\r\x1b[2C") {
		t.Errorf("cursor not left after the wrapped character: %q", out)
	}
	e.pos = 0
	out = e.render()
	if !strings.HasPrefix(out, "\x1b[1A") || !strings.HasSuffix(out, "\x1b[1A\r\x1b[2C") || e.row != 0 {
		t.Errorf("redraw from row 1 with the cursor after the prompt: %q, row %d", out, e.row)
	}
}
