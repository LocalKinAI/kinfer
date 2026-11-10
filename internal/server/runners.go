package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/LocalKinAI/kinfer/internal/mlxasr"
	"github.com/LocalKinAI/kinfer/internal/mlxengine"
	"github.com/LocalKinAI/kinfer/internal/mlxtts"
	"github.com/LocalKinAI/kinfer/internal/store"
)

// ─── one entry: speech in processes of its own ───────────────────────────────
//
// One kinfer serves the chat model and, behind the same address, speech: a
// request to /v1/audio/* goes to a speech model loaded in a kinfer process of
// its own — a runner, as Ollama runs each model — so that the chat model stays
// loaded, the three can be used at once, and a speech model that crashes takes
// only its runner down. A runner is this binary's `serve -speech off` on a
// port of the loopback, started on the first request for its model and
// stopped once idle as long as -keepalive. The model is the one the request
// names, if it is a speech model of the right kind; any other name (OpenAI's
// whisper-1 and tts-1, "tts") means this machine's default of that kind, which
// is fetched the first time if it is missing — the request is answered 503
// with how far that has got, and asked again.

// The speech modes, -speech.
const (
	SpeechAuto = "auto" // speech models start in runners on first use, stop when idle
	SpeechKeep = "keep" // the default speech models start with the server and stay
	SpeechOff  = "off"  // speech is served in this process: a speech server, or a runner
)

// runnerHeader names the model a runner is to use, as the front server
// resolved it: the request's own "model" may be an OpenAI name.
const runnerHeader = "X-Kinfer-Model"

// The kinds of speech model.
const (
	kindASR = "asr" // /v1/audio/transcriptions and translations
	kindTTS = "tts" // /v1/audio/speech and voices
)

// SpeechModel is a default speech model: what `kinfer pull` fetches, the
// store's name for it once it is here, and about how large it is.
type SpeechModel struct {
	Kind string // "asr" or "tts"
	Repo string
	Name string
	Size string
}

// SpeechModels are this machine's default speech models, each kind's in the
// order it prefers them: on a Mac whose MLX kinfer can use, Qwen3-ASR and
// Qwen3-TTS on MLX; Qwen3-ASR's GGUF everywhere. There is no text to speech
// but on MLX.
func SpeechModels() []SpeechModel {
	mlx := runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" && mlxengine.Usable() == nil
	var out []SpeechModel
	if mlx {
		out = append(out, SpeechModel{kindASR, "mlx-community/Qwen3-ASR-1.7B-8bit", "Qwen3-ASR-1.7B-8bit", "2.5 GB"})
	}
	out = append(out, SpeechModel{kindASR, "ggml-org/Qwen3-ASR-1.7B-GGUF:Q8_0", "Qwen3-ASR-1.7B-Q8_0", "2.6 GB"})
	if mlx {
		out = append(out, SpeechModel{kindTTS, "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", "1.8 GB"})
	}
	return out
}

// SetSpeech sets how speech is served (SpeechAuto, SpeechKeep or SpeechOff)
// and whether a missing default speech model is fetched on first use. addr
// is this server's own, which a runner sends its translations to. Call
// before serving.
func (s *Server) SetSpeech(mode string, autoPull bool, addr string) {
	s.speech, s.autoPull = mode, autoPull
	s.runners.keep = mode == SpeechKeep
	s.selfURL = loopbackURL(addr)
}

// SetRunner makes this server a runner for another: it takes the model a
// request names from runnerHeader.
func (s *Server) SetRunner() { s.runnerMode = true }

// runnerModel is the model a request asks for: the front server's choice,
// when this is its runner.
func (s *Server) runnerModel(r *http.Request, asked string) string {
	if s.runnerMode {
		if m := r.Header.Get(runnerHeader); m != "" {
			return m
		}
	}
	return asked
}

// loopbackURL is addr as this machine reaches it: ":11500" and
// "0.0.0.0:11500" are http://127.0.0.1:11500.
func loopbackURL(addr string) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return ""
	}
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	return "http://" + net.JoinHostPort(host, port)
}

// fits says whether the model at path is a speech model of kind: an MLX
// directory mlxasr or mlxtts serves, or a GGUF whose projector hears.
func (s *Server) fits(kind, path string) bool {
	st, err := os.Stat(path)
	if err != nil {
		return false
	}
	if st.IsDir() {
		switch kind {
		case kindASR:
			return mlxasr.IsModelDir(path)
		case kindTTS:
			return mlxtts.IsModelDir(path)
		}
		return false
	}
	if kind != kindASR {
		return false
	}
	proj := s.store.ProjectorFor(path)
	if proj == "" {
		return false
	}
	sh, err := store.ReadShape(proj)
	return err == nil && sh.HearsAudio
}

// speechModelFor is the model a speech request of kind goes to: the one it
// names, if that is a speech model of the kind; else the first default of
// the kind this store has. With none here, fetch is the default to fetch,
// or nil where the kind has none on this machine.
func (s *Server) speechModelFor(kind, asked string) (path string, fetch *SpeechModel) {
	if asked != "" {
		if p, err := s.store.Resolve(asked); err == nil && s.fits(kind, p) {
			return p, nil
		}
	}
	var first *SpeechModel
	for i := range s.speechDefaults {
		d := &s.speechDefaults[i]
		if d.Kind != kind {
			continue
		}
		if first == nil {
			first = d
		}
		if p, err := s.store.Resolve(d.Name); err == nil && modelName(p) == d.Name && s.fits(kind, p) {
			return p, nil
		}
	}
	return "", first
}

// kindNames say what a kind of speech model does, for messages.
var kindNames = map[string]string{kindASR: "speech recognition", kindTTS: "text to speech"}

// proxySpeech answers a speech request in a runner: the model it names or
// the default, fetched first if it must be. body is the request's, read.
func (s *Server) proxySpeech(w http.ResponseWriter, r *http.Request, kind, asked string, body []byte) {
	path, fetch := s.speechModelFor(kind, asked)
	if path == "" {
		switch {
		case fetch == nil:
			writeJSON(w, http.StatusNotImplemented, map[string]any{"error": map[string]string{
				"message": fmt.Sprintf("%s runs on a Mac with Apple silicon here (Qwen3-TTS on MLX); this machine has no %s model", kindNames[kind], kindNames[kind]),
				"type":    "invalid_request_error"}})
		case !s.autoPull:
			writeJSON(w, http.StatusNotFound, map[string]any{"error": map[string]string{
				"message": fmt.Sprintf("%s needs a model: kinfer pull %s (%s), or kinfer pull speech for the defaults", kindNames[kind], fetch.Repo, fetch.Size),
				"type":    "invalid_request_error"}})
		default:
			s.fetching(w, s.pulls.get(*fetch), kind)
		}
		return
	}
	rn, release, err := s.runners.acquire(path)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]string{
			"message": fmt.Sprintf("the %s model %s could not be started: %v", kindNames[kind], modelName(path), err)}})
		return
	}
	defer release()
	target, _ := url.Parse(rn.p.base)
	proxy := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(target)
			pr.Out.Header.Set(runnerHeader, path)
			if body != nil {
				pr.Out.Body = io.NopCloser(bytes.NewReader(body))
				pr.Out.ContentLength = int64(len(body))
			}
		},
		FlushInterval: -1, // streamed speech and transcripts go out as they come
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			if errors.Is(err, context.Canceled) {
				return
			}
			log.Printf("speech: %s's runner: %v", modelName(path), err)
			writeJSON(w, http.StatusBadGateway, map[string]any{"error": map[string]string{
				"message": fmt.Sprintf("the %s model %s stopped answering: %v", kindNames[kind], modelName(path), err)}})
		},
	}
	proxy.ServeHTTP(w, r)
}

// fetching answers a request while its model is fetched: 503, how far the
// fetch has got, and when to ask again; or what stopped it.
func (s *Server) fetching(w http.ResponseWriter, p *pull, kind string) {
	done, total := p.done.Load(), p.total.Load()
	select {
	case <-p.finished:
		if p.err != nil {
			writeJSON(w, http.StatusBadGateway, map[string]any{"error": map[string]string{
				"message": fmt.Sprintf("fetching %s for %s failed: %v; asking again tries again", p.model.Repo, kindNames[kind], p.err)}})
			return
		}
	default:
	}
	wait := 10
	elapsed := time.Since(p.started).Seconds()
	msg := fmt.Sprintf("%s is being fetched for %s (%s), the first time it is asked for", p.model.Name, kindNames[kind], p.model.Size)
	if total > 0 {
		msg += fmt.Sprintf(": %s of %s (%.0f%%)", store.HumanSize(done), store.HumanSize(total), float64(done)/float64(total)*100)
		if done > 0 && elapsed > 1 {
			wait = int(float64(total-done) / (float64(done) / elapsed))
		}
	}
	wait = min(max(wait, 5), 300)
	msg += fmt.Sprintf("; ask again in about %d s", wait)
	w.Header().Set("Retry-After", fmt.Sprint(wait))
	writeJSON(w, http.StatusServiceUnavailable, map[string]any{
		"error":    map[string]string{"message": msg, "type": "model_loading"},
		"progress": map[string]any{"model": p.model.Name, "repo": p.model.Repo, "done": done, "total": total},
	})
}

// speechBody reads a speech request's body and the model it names: the
// "model" field of a multipart form or of JSON. It answers the request
// itself, and returns false, for one it cannot read.
func speechBody(w http.ResponseWriter, r *http.Request, limit int64) ([]byte, string, bool) {
	if r.Body == nil {
		return nil, "", true
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if err != nil {
		transcriptionError(w, http.StatusBadRequest, "reading the request: "+err.Error())
		return nil, "", false
	}
	if int64(len(body)) > limit {
		transcriptionError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("the request is larger than %d MiB", limit>>20))
		return nil, "", false
	}
	mt, params, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if strings.HasPrefix(mt, "multipart/") {
		mr := multipart.NewReader(bytes.NewReader(body), params["boundary"])
		for {
			p, err := mr.NextPart()
			if err != nil {
				break
			}
			if p.FormName() == "model" {
				v, _ := io.ReadAll(io.LimitReader(p, 4096))
				return body, strings.TrimSpace(string(v)), true
			}
		}
		return body, "", true
	}
	var j struct {
		Model string `json:"model"`
	}
	json.Unmarshal(body, &j)
	return body, j.Model, true
}

// ─── runners ─────────────────────────────────────────────────────────────────

// proc is a runner's process: where it listens, and how to end it.
type proc struct {
	base string        // http://127.0.0.1:port
	done chan struct{} // closed when it has exited
	stop func()        // asks it to exit, then makes it
}

func (p *proc) exited() bool {
	select {
	case <-p.done:
		return true
	default:
		return false
	}
}

type runner struct {
	path     string
	p        *proc
	ready    chan struct{} // closed once it is up, or could not start
	err      error
	inflight int
	idle     *time.Timer
	expiry   time.Time
}

// runnerSet is the runners, one a model, each started on the first request
// for it and stopped once idle for keepAlive (never, with keep or 0).
type runnerSet struct {
	mu        sync.Mutex
	byPath    map[string]*runner
	keepAlive time.Duration
	keep      bool
	closed    bool
	start     func(path string) (*proc, error) // starts one and waits until it answers; tests replace it
}

// acquire is the runner for path, started if it must be; release ends the
// request's hold on it.
func (rs *runnerSet) acquire(path string) (*runner, func(), error) {
	rs.mu.Lock()
	for {
		if rs.closed {
			rs.mu.Unlock()
			return nil, nil, errors.New("the server is shutting down")
		}
		r := rs.byPath[path]
		if r == nil {
			r = &runner{path: path, ready: make(chan struct{})}
			rs.byPath[path] = r
			rs.mu.Unlock()
			p, err := rs.start(path)
			rs.mu.Lock()
			r.p, r.err = p, err
			close(r.ready)
			if err != nil {
				delete(rs.byPath, path)
				rs.mu.Unlock()
				return nil, nil, err
			}
			go rs.watch(r)
		}
		select {
		case <-r.ready:
		default:
			rs.mu.Unlock()
			<-r.ready
			rs.mu.Lock()
			continue // it may have failed, or gone: look again
		}
		if r.err != nil || r.p.exited() {
			if rs.byPath[path] == r {
				delete(rs.byPath, path)
			}
			continue
		}
		r.inflight++
		if r.idle != nil {
			r.idle.Stop()
			r.idle = nil
		}
		rs.mu.Unlock()
		var once sync.Once
		return r, func() { once.Do(func() { rs.release(r) }) }, nil
	}
}

func (rs *runnerSet) release(r *runner) {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	r.inflight--
	if r.inflight > 0 || rs.keep || rs.keepAlive <= 0 || rs.byPath[r.path] != r {
		return
	}
	r.expiry = time.Now().Add(rs.keepAlive)
	r.idle = time.AfterFunc(rs.keepAlive, func() { rs.retire(r) })
}

// retire stops r if it is still idle.
func (rs *runnerSet) retire(r *runner) {
	rs.mu.Lock()
	if rs.byPath[r.path] != r || r.inflight > 0 {
		rs.mu.Unlock()
		return
	}
	delete(rs.byPath, r.path)
	rs.mu.Unlock()
	log.Printf("speech: %s idle for %v, its runner stopped", modelName(r.path), rs.keepAlive)
	r.p.stop()
}

// watch forgets r once its process is gone, so the next request starts it
// again.
func (rs *runnerSet) watch(r *runner) {
	<-r.p.done
	rs.mu.Lock()
	gone := rs.byPath[r.path] == r
	if gone {
		delete(rs.byPath, r.path)
		if r.idle != nil {
			r.idle.Stop()
		}
	}
	rs.mu.Unlock()
	if gone {
		log.Printf("speech: %s's runner exited; the next request starts it again", modelName(r.path))
	}
}

// loaded are the runners up, for /api/ps.
func (rs *runnerSet) loaded() []*runner {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	var out []*runner
	for _, r := range rs.byPath {
		select {
		case <-r.ready:
			if r.err == nil && !r.p.exited() {
				out = append(out, r)
			}
		default:
		}
	}
	return out
}

// close stops every runner, waiting for each to exit.
func (rs *runnerSet) close() {
	rs.mu.Lock()
	rs.closed = true
	var all []*runner
	for _, r := range rs.byPath {
		all = append(all, r)
	}
	rs.byPath = map[string]*runner{}
	rs.mu.Unlock()
	var wg sync.WaitGroup
	for _, r := range all {
		<-r.ready
		if r.p == nil {
			continue
		}
		wg.Add(1)
		go func() { defer wg.Done(); r.p.stop() }()
	}
	wg.Wait()
}

// logLine is one of kinfer's own log lines: log's date and time first.
func logLine(line string) bool {
	if len(line) < len("2006/01/02 15:04:05 ") {
		return false
	}
	_, err := time.Parse("2006/01/02 15:04:05", line[:19])
	return err == nil && line[19] == ' '
}

// runnerArgs are a runner's serve flags: speech in its own process, the model
// kept until the front server stops it, and the hand sizing a speech model on
// llama.cpp wants (one slot of 16384 tokens, no prefix pool) — MLX's ignore
// them.
func runnerArgs(port string) []string {
	return []string{"serve", "-addr", "127.0.0.1:" + port, "-speech", SpeechOff, "-keepalive", "0",
		"-slots", "1", "-ctx", "16384", "-prefix", "off"}
}

// startRunner starts this binary as a runner for the model at path and waits
// until it answers. It reads its stdin and exits when that closes, so it ends
// with the front server however that ends. Its log goes to the front
// server's, each line marked, once it is up.
func (s *Server) startRunner(path string) (*proc, error) {
	exe, err := os.Executable()
	if err != nil {
		return nil, err
	}
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	_, port, _ := net.SplitHostPort(l.Addr().String())
	l.Close()

	cmd := exec.Command(exe, runnerArgs(port)...)
	cmd.Env = append(os.Environ(), "KINFER_RUNNER=1")
	if os.Getenv("KINFER_TRANSLATE_URL") == "" && s.selfURL != "" {
		cmd.Env = append(cmd.Env, "KINFER_TRANSLATE_URL="+s.selfURL) // translations by this server's chat model
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	pr, pw := io.Pipe()
	cmd.Stdout, cmd.Stderr = pw, pw
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	name := modelName(path)
	// What the runner says: kinfer's own log lines go to this server's log
	// once it is up; llama.cpp's and Metal's chatter, and the banner, only
	// into the last lines kept, which are told should it end unasked.
	var up, stopping atomic.Bool
	var tail []string
	var tailMu sync.Mutex
	lines := make(chan struct{})
	go func() {
		defer close(lines)
		sc := bufio.NewScanner(pr)
		sc.Buffer(nil, 1<<20)
		for sc.Scan() {
			line := sc.Text()
			if strings.TrimSpace(line) == "" {
				continue
			}
			tailMu.Lock()
			tail = append(tail, line)
			if len(tail) > 20 {
				tail = tail[1:]
			}
			tailMu.Unlock()
			if up.Load() && logLine(line) {
				log.Printf("speech %s: %s", name, line[len("2006/01/02 15:04:05 "):])
			}
		}
	}()
	last := func() string {
		tailMu.Lock()
		defer tailMu.Unlock()
		return strings.Join(tail, "\n  ")
	}
	done := make(chan struct{})
	go func() {
		err := cmd.Wait()
		pw.Close()
		<-lines
		if up.Load() && !stopping.Load() {
			log.Printf("speech: %s's runner ended (%v); what it said last:\n  %s", name, err, last())
		}
		close(done)
	}()
	p := &proc{base: "http://127.0.0.1:" + port, done: done}
	p.stop = func() {
		stopping.Store(true)
		stdin.Close()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			cmd.Process.Kill()
			<-done
		}
	}

	t0 := time.Now()
	client := &http.Client{Timeout: time.Second}
	for {
		if resp, err := client.Get(p.base + "/api/version"); err == nil {
			resp.Body.Close()
			up.Store(true)
			log.Printf("speech: %s's runner up on %s in %v", name, p.base, time.Since(t0).Round(time.Millisecond))
			return p, nil
		}
		select {
		case <-done:
			return nil, fmt.Errorf("its runner exited:\n  %s", last())
		case <-time.After(100 * time.Millisecond):
		}
		if time.Since(t0) > time.Minute {
			p.stop()
			return nil, errors.New("its runner did not answer within a minute")
		}
	}
}

// ─── fetching a default speech model ─────────────────────────────────────────

type pull struct {
	model       SpeechModel
	done, total atomic.Int64
	started     time.Time
	finished    chan struct{}
	err         error
	ended       time.Time
}

// pullSet is the default speech models being fetched, one at a time each.
type pullSet struct {
	mu     sync.Mutex
	byRepo map[string]*pull
	start  func(p *pull) // fetches p.model, recording progress, and closes p.finished; tests replace it
	procs  []*exec.Cmd
}

// get is the fetch of m under way, or a new one; one that failed is tried
// again after half a minute.
func (ps *pullSet) get(m SpeechModel) *pull {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if p := ps.byRepo[m.Repo]; p != nil {
		select {
		case <-p.finished:
			if p.err == nil || time.Since(p.ended) < 30*time.Second {
				return p
			}
		default:
			return p
		}
	}
	p := &pull{model: m, started: time.Now(), finished: make(chan struct{})}
	ps.byRepo[m.Repo] = p
	log.Printf("speech: fetching %s (%s) for its first use; -auto-pull=false stops this", m.Repo, m.Size)
	go ps.start(p)
	return p
}

// runPull fetches p.model with this binary's `kinfer pull`, which says how
// far it has got as JSON lines (KINFER_PULL_PROGRESS=json).
func (ps *pullSet) runPull(p *pull) {
	defer close(p.finished)
	defer func() { p.ended = time.Now() }()
	exe, err := os.Executable()
	if err != nil {
		p.err = err
		return
	}
	cmd := exec.Command(exe, "pull", p.model.Repo)
	cmd.Env = append(os.Environ(), "KINFER_PULL_PROGRESS=json")
	out, err := cmd.StdoutPipe()
	if err != nil {
		p.err = err
		return
	}
	var errBuf bytes.Buffer
	cmd.Stderr = &errBuf
	if err := cmd.Start(); err != nil {
		p.err = err
		return
	}
	ps.mu.Lock()
	ps.procs = append(ps.procs, cmd)
	ps.mu.Unlock()
	last := ""
	sc := bufio.NewScanner(out)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "{") {
			var st struct{ Done, Total int64 }
			if json.Unmarshal([]byte(line), &st) == nil {
				p.done.Store(st.Done)
				p.total.Store(st.Total)
			}
			continue
		}
		if strings.TrimSpace(line) != "" {
			last = strings.TrimSpace(line)
			log.Printf("speech pull %s: %s", p.model.Name, last)
		}
	}
	if err := cmd.Wait(); err != nil {
		said := strings.TrimSpace(errBuf.String())
		if said == "" {
			said = last
		}
		p.err = fmt.Errorf("%v: %s", err, said)
		log.Printf("speech: fetching %s failed: %v", p.model.Repo, p.err)
		return
	}
	log.Printf("speech: %s is here (%s)", p.model.Name, filepath.Base(p.model.Repo))
}

// stop ends the fetches under way.
func (ps *pullSet) stop() {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	for _, c := range ps.procs {
		if c.Process != nil && c.ProcessState == nil {
			c.Process.Kill()
		}
	}
}

// PrepareSpeech readies the default speech models with -speech keep: each
// fetched if it must be and may be, then started in its runner, to stay.
func (s *Server) PrepareSpeech() {
	for _, kind := range []string{kindASR, kindTTS} {
		path, fetch := s.speechModelFor(kind, "")
		if path == "" && fetch != nil && s.autoPull {
			p := s.pulls.get(*fetch)
			<-p.finished
			path, _ = s.speechModelFor(kind, "")
		}
		if path == "" {
			continue
		}
		rn, release, err := s.runners.acquire(path)
		if err != nil {
			log.Printf("speech: %s could not be started: %v", modelName(path), err)
			continue
		}
		t0 := time.Now()
		if err := warm(rn.p.base, kind, path); err != nil {
			log.Printf("speech: %s did not load: %v", modelName(path), err)
		} else {
			log.Printf("speech: %s loaded in %v, kept", modelName(path), time.Since(t0).Round(time.Millisecond))
		}
		release()
	}
}

// warm has a runner load its model, so that the first request finds it
// loaded: the voices of a text to speech model, a transcript of half a
// second of silence from a speech recognition one.
func warm(base, kind, path string) error {
	var req *http.Request
	var err error
	if kind == kindTTS {
		req, err = http.NewRequest("GET", base+"/v1/audio/voices?model=x", nil)
	} else {
		var b bytes.Buffer
		mw := multipart.NewWriter(&b)
		mw.WriteField("model", "x")
		fw, _ := mw.CreateFormFile("file", "silence.wav")
		fw.Write(append(wavHeader(16000, 16000), make([]byte, 16000)...)) // half a second
		mw.Close()
		req, err = http.NewRequest("POST", base+"/v1/audio/transcriptions", &b)
		if err == nil {
			req.Header.Set("Content-Type", mw.FormDataContentType())
		}
	}
	if err != nil {
		return err
	}
	req.Header.Set(runnerHeader, path)
	resp, err := (&http.Client{Timeout: 5 * time.Minute}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s", resp.Status)
	}
	return nil
}
