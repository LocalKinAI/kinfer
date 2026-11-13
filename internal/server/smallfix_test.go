package server

import (
	"bufio"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/engine"
)

// A panic while a model loads is a failed load, not the end of the process
// and of every other request in it.
func TestAPanicWhileLoadingIsAFailedLoad(t *testing.T) {
	g, err := openSafely(func(string, engine.Options) (generator, error) { panic("bad tensor") }, "/m/x.gguf", engine.Options{})
	if g != nil || err == nil || !strings.Contains(err.Error(), "bad tensor") {
		t.Fatalf("a panic while loading: %v %v", g, err)
	}
	want := errors.New("no such file")
	if _, err := openSafely(func(string, engine.Options) (generator, error) { return nil, want }, "x", engine.Options{}); err != want {
		t.Errorf("an error while loading came back as %v", err)
	}
}

// What a voice is told in words is bounded as what it says is.
func TestSpeechInstructionsAreBounded(t *testing.T) {
	srv, _ := newTestServer(t, "tts")
	body := `{"model":"tts","input":"hi","instructions":"` + strings.Repeat("a", maxSpeechInput+1) + `"}`
	req := httptest.NewRequest("POST", "/v1/audio/speech", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.handleSpeech(w, req)
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "instructions") {
		t.Errorf("instructions of %d characters: %d %s", maxSpeechInput+1, w.Code, w.Body.String())
	}
}

// A runner's line over the limit is cut, the rest of it read and dropped, and
// the lines after it read: the Scanner stopped there, and the runner hung on
// its next write into a pipe nobody read.
func TestALongLogLineDoesNotStopTheReading(t *testing.T) {
	long := strings.Repeat("x", 3<<20)
	rd := bufio.NewReaderSize(strings.NewReader("first\n"+long+"\n2026/11/13 10:00:00 after\nlast"), 64<<10)
	var got []string
	for {
		line, err := readLogLine(rd, 1<<20)
		if err != nil && line == "" {
			break
		}
		got = append(got, line)
	}
	if len(got) != 4 || got[0] != "first" || len(got[1]) != 1<<20 || got[2] != "2026/11/13 10:00:00 after" || got[3] != "last" {
		t.Fatalf("read %d lines: %q %d %q", len(got), got[0], len(got[1]), got[2:])
	}
}
