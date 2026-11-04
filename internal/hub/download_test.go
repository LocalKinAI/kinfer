package hub

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/LocalKinAI/kinfer/internal/store"
)

// fakeHub serves one file the way Hugging Face does: its resolve address
// redirects to a CDN address that is signed (here, numbered) and serves
// ranges.
type fakeHub struct {
	file     atomic.Pointer[blob]
	resolves atomic.Int64
	signed   atomic.Int64 // the number of the latest address handed out
	expired  atomic.Int64 // addresses numbered up to this one answer 403

	noRanges bool
	cut      int64 // each CDN reply drops its connection after this many bytes; 0 never

	stopAt   atomic.Int64 // once this many bytes have gone out, call stop
	stop     func()
	expireAt atomic.Int64 // once this many bytes have gone out, expire every address so far

	refuse    atomic.Int64 // the CDN answers this many requests 503 before serving
	downAfter atomic.Int64 // once this many bytes have gone out, the CDN answers 503
	delay     time.Duration
	cdn       string // where the resolve address redirects to; this server when empty

	mu    sync.Mutex
	bySig map[int64]int64 // bytes sent in replies to each address
}

type blob struct {
	data []byte
	sum  string // the sha256 the resolve answer publishes
}

func newBlob(size int) *blob {
	data := make([]byte, size)
	rand.Read(data)
	sum := sha256.Sum256(data)
	return &blob{data: data, sum: hex.EncodeToString(sum[:])}
}

func serve(t *testing.T, size int) *fakeHub {
	h := &fakeHub{bySig: map[int64]int64{}}
	h.file.Store(newBlob(size))
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	t.Setenv("KINFER_HF_ENDPOINT", srv.URL)
	return h
}

func (h *fakeHub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b := h.file.Load()
	if strings.Contains(r.URL.Path, "/resolve/") {
		h.resolves.Add(1)
		w.Header().Set("X-Linked-Size", strconv.Itoa(len(b.data)))
		w.Header().Set("X-Linked-Etag", `"`+b.sum+`"`)
		w.Header().Set("X-Repo-Commit", "c0ffee")
		http.Redirect(w, r, fmt.Sprintf("%s/cdn/blob?sig=%d", h.cdn, h.signed.Add(1)), http.StatusFound)
		return
	}
	sig, _ := strconv.ParseInt(r.URL.Query().Get("sig"), 10, 64)
	if sig <= h.expired.Load() {
		http.Error(w, "expired", http.StatusForbidden)
		return
	}
	if h.refuse.Add(-1) >= 0 {
		http.Error(w, "busy", http.StatusServiceUnavailable)
		return
	}
	if at := h.downAfter.Load(); at > 0 && h.served(0) >= at {
		http.Error(w, "down", http.StatusServiceUnavailable)
		return
	}
	if h.noRanges {
		r.Header.Del("Range")
	}
	http.ServeContent(&cdnWriter{ResponseWriter: w, hub: h, sig: sig, left: h.cut},
		r, "blob", time.Time{}, bytes.NewReader(b.data))
}

// served is how many bytes went out in replies to addresses numbered above
// after.
func (h *fakeHub) served(after int64) int64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	var n int64
	for sig, sent := range h.bySig {
		if sig > after {
			n += sent
		}
	}
	return n
}

type cdnWriter struct {
	http.ResponseWriter
	hub  *fakeHub
	sig  int64
	left int64
}

func (w *cdnWriter) Write(b []byte) (int, error) {
	time.Sleep(w.hub.delay)
	drop := w.left > 0 && int64(len(b)) >= w.left
	if drop {
		b = b[:w.left]
	}
	n, err := w.ResponseWriter.Write(b)
	w.left -= int64(n)

	h := w.hub
	h.mu.Lock()
	h.bySig[w.sig] += int64(n)
	h.mu.Unlock()
	total := h.served(0)
	if at := h.stopAt.Load(); at > 0 && total >= at {
		h.stop()
	}
	if at := h.expireAt.Load(); at > 0 && total >= at {
		h.expired.Store(h.signed.Load())
		h.expireAt.Store(0)
	}
	if drop {
		w.ResponseWriter.(http.Flusher).Flush()
		panic(http.ErrAbortHandler) // the connection drops mid-reply
	}
	return n, err
}

// small makes a download of a megabyte take sixteen pieces, and its retries
// take no time.
func small(t *testing.T) {
	ps, rb, pa := pieceSize, retryBase, patience
	pieceSize, retryBase, patience = 64<<10, time.Millisecond, 10*time.Second
	t.Cleanup(func() { pieceSize, retryBase, patience = ps, rb, pa })
}

func holds(t *testing.T, path string, want []byte) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s holds %d bytes that are not the %d served", path, len(got), len(want))
	}
}

// only fails unless dir holds exactly the named files.
func only(t *testing.T, dir string, names ...string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, e := range entries {
		got = append(got, e.Name())
	}
	if strings.Join(got, " ") != strings.Join(names, " ") {
		t.Errorf("%s holds %q, want %q", dir, got, names)
	}
}

func TestDownloadInPieces(t *testing.T) {
	small(t)
	h := serve(t, 1<<20+123) // sixteen whole pieces and a short one
	dir := t.TempDir()

	var last Status
	path, err := Download(context.Background(), "o/r", "m.gguf", dir, 4, func(s Status) { last = s })
	if err != nil {
		t.Fatal(err)
	}
	holds(t, path, h.file.Load().data)
	if size := int64(1<<20 + 123); last.Done != size || last.Checked != size || last.Kept != 0 {
		t.Errorf("last status %+v, want %d bytes done and checked, none kept", last, size)
	}
	if got := h.served(0); got != 1<<20+123 {
		t.Errorf("the CDN sent %d bytes of a %d-byte file", got, 1<<20+123)
	}
	only(t, dir, "m.gguf")
}

// A pull that stops keeps what it fetched, and the next fetches only the rest
// — in the pieces the first cut the file into, whatever the size is now.
func TestDownloadGoesOnAfterStop(t *testing.T) {
	small(t)
	h := serve(t, 1<<20)
	size := int64(1 << 20)
	dir := t.TempDir()

	ctx, stop := context.WithCancel(context.Background())
	h.stop = stop
	h.stopAt.Store(size * 2 / 5)
	_, err := Download(ctx, "o/r", "m.gguf", dir, 4, nil)
	if err == nil || !strings.Contains(err.Error(), "goes on from there") {
		t.Fatalf("a stopped pull returned %v", err)
	}
	h.stopAt.Store(0)
	only(t, dir, "m.gguf.part", "m.gguf.part.json")

	pieceSize = 128 << 10
	first := h.signed.Load()
	var kept int64 = -1
	path, err := Download(context.Background(), "o/r", "m.gguf", dir, 4, func(s Status) {
		if kept < 0 {
			kept = s.Kept
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	holds(t, path, h.file.Load().data)
	if kept <= 0 || kept >= size {
		t.Fatalf("the second pull found %d bytes kept of %d", kept, size)
	}
	if got := h.served(first); got != size-kept {
		t.Errorf("the second pull fetched %d bytes; the first left %d to fetch", got, size-kept)
	}
	only(t, dir, "m.gguf")
}

// A file that changed upstream since a pull stopped is fetched from the start.
func TestDownloadStartsOverWhenFileChanged(t *testing.T) {
	small(t)
	h := serve(t, 1<<20)
	dir := t.TempDir()

	ctx, stop := context.WithCancel(context.Background())
	h.stop = stop
	h.stopAt.Store(400 << 10)
	if _, err := Download(ctx, "o/r", "m.gguf", dir, 4, nil); err == nil {
		t.Fatal("the pull should have stopped")
	}
	h.stopAt.Store(0)

	h.file.Store(newBlob(1 << 20))
	var kept int64 = -1
	path, err := Download(context.Background(), "o/r", "m.gguf", dir, 4, func(s Status) {
		if kept < 0 {
			kept = s.Kept
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	holds(t, path, h.file.Load().data)
	if kept != 0 {
		t.Errorf("kept %d bytes of the file's earlier version", kept)
	}
}

// A connection that drops mid-reply costs only what had not arrived.
func TestDownloadSurvivesDroppedConnections(t *testing.T) {
	small(t)
	h := serve(t, 1<<20)
	h.cut = 40 << 10 // every reply drops after 40 KB of a 64 KB piece
	path, err := Download(context.Background(), "o/r", "m.gguf", t.TempDir(), 4, nil)
	if err != nil {
		t.Fatal(err)
	}
	holds(t, path, h.file.Load().data)
}

// The CDN's address expires after an hour; a pull taking longer asks for a new
// one and goes on.
func TestDownloadRenewsExpiredAddress(t *testing.T) {
	small(t)
	h := serve(t, 1<<20)
	h.expireAt.Store(300 << 10)
	path, err := Download(context.Background(), "o/r", "m.gguf", t.TempDir(), 4, nil)
	if err != nil {
		t.Fatal(err)
	}
	holds(t, path, h.file.Load().data)
	if n := h.resolves.Load(); n < 2 {
		t.Errorf("resolved the file %d times; an expired address needs a second", n)
	}
}

func TestDownloadRejectsWrongSum(t *testing.T) {
	small(t)
	h := serve(t, 256<<10)
	b := h.file.Load()
	h.file.Store(&blob{data: b.data, sum: strings.Repeat("0", 64)})
	dir := t.TempDir()

	_, err := Download(context.Background(), "o/r", "m.gguf", dir, 4, nil)
	if !errors.Is(err, errMismatch) {
		t.Fatalf("a file that does not match its sha256 returned %v", err)
	}
	only(t, dir)
}

func TestDownloadWithoutRanges(t *testing.T) {
	small(t)
	h := serve(t, 300<<10)
	h.noRanges = true
	dir := t.TempDir()
	path, err := Download(context.Background(), "o/r", "m.gguf", dir, 4, nil)
	if err != nil {
		t.Fatal(err)
	}
	holds(t, path, h.file.Load().data)
	only(t, dir, "m.gguf")
}

func TestDownloadSkipsFileAlreadyHere(t *testing.T) {
	h := serve(t, 256<<10)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "m.gguf"), h.file.Load().data, 0o644); err != nil {
		t.Fatal(err)
	}
	var last Status
	if _, err := Download(context.Background(), "o/r", "m.gguf", dir, 4, func(s Status) { last = s }); err != nil {
		t.Fatal(err)
	}
	if got := h.served(0); got != 0 {
		t.Errorf("fetched %d bytes of a file already here", got)
	}
	if last.Kept != last.Total || last.Total != 256<<10 {
		t.Errorf("status %+v, want everything kept", last)
	}
}

func TestDownloadRefusesSecondPullOfSameFile(t *testing.T) {
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("no file locks here")
	}
	serve(t, 256<<10)
	dir := t.TempDir()
	f, err := os.OpenFile(filepath.Join(dir, "m.gguf.part"), os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := lock(f); err != nil {
		t.Fatal(err)
	}
	if _, err := Download(context.Background(), "o/r", "m.gguf", dir, 4, nil); err == nil || !strings.Contains(err.Error(), "already") {
		t.Fatalf("a second pull of a file being pulled returned %v", err)
	}
}

// Each piece needs a TCP connection of its own, so a download must speak
// HTTP/1.1 even to a server that offers HTTP/2 — and even after
// http.DefaultClient has, which leaves http.DefaultTransport offering it.
func TestDownloadConnectionsAreHTTP1(t *testing.T) {
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, r.Proto)
	}))
	srv.EnableHTTP2 = true
	srv.StartTLS()
	defer srv.Close()
	if resp, err := http.Get(srv.URL); err == nil { // fails on the test certificate, after the TLS settings are made
		resp.Body.Close()
	}

	tr := newClients(4).fetcher.Transport.(*http.Transport)
	if tr.TLSClientConfig == nil {
		tr.TLSClientConfig = &tls.Config{}
	}
	tr.TLSClientConfig.RootCAs = srv.Client().Transport.(*http.Transport).TLSClientConfig.RootCAs
	resp, err := (&http.Client{Transport: tr}).Get(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.ProtoMajor != 1 {
		t.Errorf("a download spoke %s; each piece needs a connection of its own", resp.Proto)
	}
}

// A CDN that turns every request away for a while is waited out, and the pull
// says why it is waiting while it does.
func TestDownloadWaitsOutAnOutage(t *testing.T) {
	small(t)
	h := serve(t, 1<<20)
	h.refuse.Store(40) // ten requests in a row on each of four connections

	var waited bool
	var last Status
	path, err := Download(context.Background(), "o/r", "m.gguf", t.TempDir(), 4, func(s Status) {
		if strings.Contains(s.Waiting, "503") {
			waited = true
		}
		last = s
	})
	if err != nil {
		t.Fatal(err)
	}
	holds(t, path, h.file.Load().data)
	if !waited {
		t.Error("no status said why the pull was waiting")
	}
	if last.Waiting != "" {
		t.Errorf("the last status still says %q, after every byte arrived", last.Waiting)
	}
}

// A CDN down for longer than patience stops the pull, which keeps what came.
func TestDownloadGivesUpAfterPatience(t *testing.T) {
	small(t)
	patience = 50 * time.Millisecond
	h := serve(t, 1<<20)
	h.downAfter.Store(300 << 10)
	dir := t.TempDir()

	_, err := Download(context.Background(), "o/r", "m.gguf", dir, 4, nil)
	if err == nil || !strings.Contains(err.Error(), "503") || !strings.Contains(err.Error(), "goes on from there") {
		t.Fatalf("a pull whose CDN went down returned %v", err)
	}
	only(t, dir, "m.gguf.part", "m.gguf.part.json")
}

// An error about a request to the CDN leaves out its address: a few hundred
// characters of signature that say nothing to the person reading it.
func TestDownloadErrorsLeaveOutTheAddress(t *testing.T) {
	small(t)
	patience = 50 * time.Millisecond
	h := serve(t, 256<<10)
	gone := httptest.NewServer(http.NotFoundHandler())
	h.cdn = gone.URL
	gone.Close() // every request to the CDN is refused

	_, err := Download(context.Background(), "o/r", "m.gguf", t.TempDir(), 4, nil)
	if err == nil || !strings.Contains(err.Error(), "refused") {
		t.Fatalf("a pull from a CDN that refuses connections returned %v", err)
	}
	if strings.Contains(err.Error(), "sig=") {
		t.Errorf("the error carries the CDN's address: %v", err)
	}
}

// fillDisk makes the disk say it has avail bytes free.
func fillDisk(t *testing.T, avail int64) {
	df := diskFree
	diskFree = func(string) (int64, error) { return avail, nil }
	t.Cleanup(func() { diskFree = df })
}

// A pull stops while the disk still has diskFloor free, and keeps what came.
func TestDownloadStopsBeforeTheDiskFills(t *testing.T) {
	small(t)
	h := serve(t, 1<<20)
	h.delay = 60 * time.Millisecond // a 32 KB write each: the pull takes half a second, and looks at the disk after 200 ms
	fillDisk(t, diskFloor-1)

	_, err := Download(context.Background(), "o/r", "m.gguf", t.TempDir(), 4, nil)
	if !errors.Is(err, errDiskFull) {
		t.Fatalf("a pull on a disk below its floor returned %v", err)
	}
	if !strings.Contains(err.Error(), "goes on from there") {
		t.Errorf("a pull stopped by the disk did not keep what came: %v", err)
	}
}

// Room counts what a pull still needs: nothing for a file already here, and
// only the rest of one an earlier pull stopped in.
func TestRoom(t *testing.T) {
	small(t)
	h := serve(t, 1<<20)
	dir := t.TempDir()

	ctx, stop := context.WithCancel(context.Background())
	h.stop = stop
	h.stopAt.Store(512 << 10)
	if _, err := Download(ctx, "o/r", "m.gguf", dir, 4, nil); err == nil {
		t.Fatal("the pull should have stopped")
	}
	need := left(filepath.Join(dir, "m.gguf"), 1<<20)
	if need <= 0 || need >= 1<<20 {
		t.Fatalf("a pull stopped halfway still needs %d bytes of %d", need, 1<<20)
	}

	files := []File{{Name: "m.gguf", Size: 1 << 20}}
	fillDisk(t, diskFloor+need) // room for the rest of it, not for the whole
	if err := Room(dir, files); err != nil {
		t.Errorf("Room did not count the bytes an earlier pull kept: %v", err)
	}
	fillDisk(t, diskFloor+need-1)
	if err := Room(dir, files); err == nil {
		t.Error("Room let a pull start that would take the disk below its floor")
	}
	if err := os.WriteFile(filepath.Join(dir, "m.gguf"), h.file.Load().data, 0o644); err != nil {
		t.Fatal(err)
	}
	fillDisk(t, 0)
	if err := Room(dir, files); err != nil {
		t.Errorf("Room refused a pull with nothing left to fetch: %v", err)
	}
}

// What a stopped pull leaves is what the store shows as unfinished, and what
// kinfer rm removes: the store reads the record download.go writes.
func TestStoppedPullIsUnfinishedInTheStore(t *testing.T) {
	small(t)
	h := serve(t, 1<<20)
	dir := t.TempDir()
	ctx, stop := context.WithCancel(context.Background())
	h.stop = stop
	h.stopAt.Store(400 << 10)
	if _, err := Download(ctx, "o/r", "m.gguf", dir, 4, nil); err == nil {
		t.Fatal("the pull should have stopped")
	}

	st, err := store.OpenAt(dir)
	if err != nil {
		t.Fatal(err)
	}
	unfinished, err := st.Unfinished()
	if err != nil {
		t.Fatal(err)
	}
	kept := 1<<20 - left(filepath.Join(dir, "m.gguf"), 1<<20)
	if len(unfinished) != 1 || unfinished[0].Name != "m" || unfinished[0].Size != kept || unfinished[0].Total != 1<<20 {
		t.Fatalf("the store shows %+v; the pull kept %d bytes of m.gguf's %d", unfinished, kept, 1<<20)
	}
	if err := st.Remove("m"); err != nil {
		t.Fatal(err)
	}
	only(t, dir)
}

// A CDN that turns away every address, new ones too, is asked again at the
// pace of any other failure — not in a loop of requests to Hugging Face.
func TestDownloadBacksOffWhenEveryAddressIsRefused(t *testing.T) {
	small(t)
	patience = 100 * time.Millisecond
	retryBase = 5 * time.Millisecond
	h := serve(t, 256<<10)
	h.expired.Store(1 << 62)

	_, err := Download(context.Background(), "o/r", "m.gguf", t.TempDir(), 4, nil)
	if err == nil {
		t.Fatal("a pull no address works for succeeded")
	}
	// Each connection: an immediate retry, then 5, 10, 20, 40 ms apart.
	if n := h.resolves.Load(); n > 40 {
		t.Errorf("asked Hugging Face for an address %d times in %s", n, patience)
	}
}

// A file downloaded to a name of the caller's lands there, and nowhere else.
func TestDownloadAs(t *testing.T) {
	small(t)
	h := serve(t, 300<<10)
	dir := t.TempDir()
	dest := filepath.Join(dir, "Vis-Q4_K_M.mmproj.gguf")
	path, err := DownloadAs(context.Background(), "o/r", "mmproj-F16.gguf", dest, 4, nil)
	if err != nil || path != dest {
		t.Fatalf("DownloadAs = %q, %v", path, err)
	}
	holds(t, dest, h.file.Load().data)
	only(t, dir, "Vis-Q4_K_M.mmproj.gguf")
}
