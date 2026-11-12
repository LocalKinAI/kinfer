package hub

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/LocalKinAI/kinfer/internal/store"
)

// Status is how far a download has got.
type Status struct {
	Done    int64 // bytes of the file on disk
	Total   int64 // the file's size; 0 when the server does not say
	Kept    int64 // of Done, what an earlier pull had fetched
	Checked int64 // bytes compared so far with the sha256 Hugging Face publishes

	// Waiting is why requests are failing, while the pull waits to ask
	// again; it is empty once bytes arrive.
	Waiting string
}

// Progress reports a download's state, a few times a second.
type Progress func(Status)

// Variables rather than constants so the tests can cut a few hundred
// kilobytes into many pieces, retry without waiting, and fill a disk.
var (
	// pieceSize is how much one request asks for: big enough that a
	// request's round trip is a small part of its time, small enough that
	// eight connections all have work in a file of a few hundred megabytes and
	// that the slowest of them holds up the end of a pull by little.
	pieceSize int64 = 32 << 20

	// retryBase is the wait before asking again for a piece. It doubles with
	// each request in a row that fetches nothing, up to 32 times this.
	retryBase = time.Second

	// idleLimit is how long a connection may deliver nothing before it is
	// dropped and its piece asked for again. A connection can stall without
	// closing, and nothing else would notice.
	idleLimit = time.Minute

	// patience is how long a piece may go without a byte, request after
	// request, before the pull stops: long enough to ride out a router
	// restarting or Wi-Fi dropping for a while. Time a Mac spends asleep does
	// not count, and a pull goes on when it wakes.
	patience = 10 * time.Minute

	// diskFloor is the space a pull leaves free on its disk. A full disk
	// fails everything on the machine at once, macOS's swap among them, and
	// what a stopped pull fetched stays there.
	diskFloor int64 = 2 << 30

	// diskFree is how the space left is read.
	diskFree = free
)

// saveEvery is how often the record of what has arrived is written, which is
// what a crash can cost.
const saveEvery = 2 * time.Second

var (
	errNoRanges  = errors.New("the server ignores ranged requests")
	errMismatch  = errors.New("the file does not match the sha256 Hugging Face publishes for it")
	errExpired   = errors.New("the CDN's address for the file expired")
	errDiskFull  = errors.New("the disk is nearly full")
	errRedirects = errors.New("stopped after 10 redirects")
)

// permanent marks an error that asking again cannot fix.
type permanent struct{ error }

func (p permanent) Unwrap() error { return p.error }

// Download fetches a file into dir and returns its path.
//
// One connection to Hugging Face's CDN carries a fraction of what a home link
// can, so a file is fetched in pieces, conns of them at once. The bytes land in
// name.part, and name.part.json records how far each piece has got: a pull that
// stops — Ctrl-C, a dropped network, a full disk, a laptop lid — leaves both,
// and the next pull of the same file fetches only what is missing. The file is
// hashed as it fills and takes its name only once every byte is there and the
// whole matches the sha256 Hugging Face publishes for it (every GGUF has one),
// so neither an interrupted pull nor a resume gone wrong can leave something
// that looks like a usable model.
func Download(ctx context.Context, repo, filename, dir string, conns int, onProgress Progress) (string, error) {
	return DownloadAs(ctx, repo, filename, filepath.Join(dir, filepath.Base(filename)), conns, onProgress)
}

// DownloadAs is Download to a name of the caller's: a projector, which
// repos all call mmproj-F16.gguf, kept as its model's NAME.mmproj.gguf.
func DownloadAs(ctx context.Context, repo, filename, final string, conns int, onProgress Progress) (string, error) {
	if onProgress == nil {
		onProgress = func(Status) {}
	}
	conns = max(conns, 1)
	c := newClients(conns)

	src := fmt.Sprintf("%s/%s/resolve/main/%s", endpoint(), repo, filename)
	r, err := c.resolveSoon(ctx, src, onProgress)
	if ctx.Err() != nil {
		return "", fmt.Errorf("%s: stopped", filename)
	}
	if err != nil {
		return "", fmt.Errorf("download %s: %w", filename, err)
	}
	if r.commit != "" {
		// Every later request names this commit, so a push to the repo
		// during the pull cannot leave pieces of two versions in one file.
		src = fmt.Sprintf("%s/%s/resolve/%s/%s", endpoint(), repo, r.commit, filename)
	}

	d := &download{clients: c, src: src, url: r.url, size: r.size, etag: r.etag}
	err = d.open(final + ".part")
	if d.file != nil {
		defer d.file.Close()
	}
	if err != nil {
		return "", fmt.Errorf("download %s: %w", filename, err)
	}
	// Looked for under the lock, so that a pull of the same file finishing a
	// moment ago is seen finished.
	if fi, err := os.Stat(final); err == nil && r.size > 0 && fi.Size() == r.size && hasSum(final, r.sha256) {
		os.Remove(d.file.Name())
		d.forget()
		onProgress(Status{Done: r.size, Total: r.size, Kept: r.size})
		return final, nil
	}
	if err := d.prepare(); err != nil {
		return "", fmt.Errorf("download %s: %w", filename, err)
	}

	err = errNoRanges
	if d.size > 0 {
		err = d.run(ctx, conns, r.sha256, onProgress)
	}
	if errors.Is(err, errNoRanges) {
		err = d.stream(ctx, r.sha256, onProgress)
	}
	if err == nil {
		err = d.finish(final)
	}
	if err != nil {
		return "", d.failed(ctx, filename, err)
	}
	if r.sha256 != "" {
		rememberSum(final, r.sha256) // the pull checked the whole against it
	}
	return final, nil
}

// sumsFile is, hidden in a directory of pulled files, the sha256 of each with
// the size and time it had when it was summed. A pull that finds its file
// already there of the right size keeps it only if its sha256 is the one the
// repo publishes: a file pushed again at the same size, or a shard of an
// earlier version beside the pieces of a later one, is fetched again. The
// record spares hashing tens of gigabytes each time a model is pulled again;
// a file it does not know, or that changed since, is hashed once.
const sumsFile = ".kinfer-sha256.json"

type summed struct {
	Size   int64  `json:"size"`
	Time   int64  `json:"mtime"` // UnixNano
	SHA256 string `json:"sha256"`
}

var sumsMu sync.Mutex

// hasSum reports whether the file at path has the sha256 want; with nothing
// published to compare with, whether it is there at all.
func hasSum(path, want string) bool {
	if want == "" {
		return true
	}
	got, err := sumOf(path)
	return err == nil && got == want
}

// sumOf is the file's sha256: as recorded, while the file is as it was when
// summed, else hashed now and recorded.
func sumOf(path string) (string, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	sumsMu.Lock()
	rec, ok := readSums(filepath.Dir(path))[filepath.Base(path)]
	sumsMu.Unlock()
	if ok && rec.Size == fi.Size() && rec.Time == fi.ModTime().UnixNano() {
		return rec.SHA256, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	sum := hex.EncodeToString(h.Sum(nil))
	rememberSum(path, sum)
	return sum, nil
}

// rememberSum records sum as the sha256 of the file at path as it is now.
func rememberSum(path, sum string) {
	fi, err := os.Stat(path)
	if err != nil {
		return
	}
	sumsMu.Lock()
	defer sumsMu.Unlock()
	dir := filepath.Dir(path)
	all := readSums(dir)
	all[filepath.Base(path)] = summed{Size: fi.Size(), Time: fi.ModTime().UnixNano(), SHA256: sum}
	b, err := json.MarshalIndent(all, "", " ")
	if err != nil {
		return
	}
	tmp := filepath.Join(dir, sumsFile+".tmp")
	if os.WriteFile(tmp, b, 0o644) == nil {
		os.Rename(tmp, filepath.Join(dir, sumsFile))
	}
}

func readSums(dir string) map[string]summed {
	all := map[string]summed{}
	if b, err := os.ReadFile(filepath.Join(dir, sumsFile)); err == nil {
		json.Unmarshal(b, &all)
	}
	return all
}

// Room says whether the disk dir is on can take what files still need and
// keep diskFloor free. A file here already needs nothing, and one an earlier
// pull stopped in only what that pull did not fetch. Sizes are as List
// reports them; each file is kept in dir under the base of its Name.
func Room(dir string, files []File) error {
	avail, err := diskFree(dir)
	if err != nil {
		return nil // nothing to go by
	}
	var need int64
	for _, f := range files {
		need += left(filepath.Join(dir, filepath.Base(f.Name)), f.Size)
	}
	if need == 0 || avail-need >= diskFloor {
		return nil
	}
	return fmt.Errorf("the pull needs %s more and the disk has %s free, of which a pull leaves %s; make room and pull again",
		store.HumanSize(need), store.HumanSize(avail), store.HumanSize(diskFloor))
}

// left is how much of a file of size bytes is still to fetch to path.
func left(path string, size int64) int64 {
	if fi, err := os.Stat(path); err == nil && fi.Size() == size {
		return 0
	}
	var rec record
	if b, err := os.ReadFile(path + ".part.json"); err == nil && json.Unmarshal(b, &rec) == nil && rec.Size == size {
		for _, done := range rec.Done {
			size -= done
		}
	}
	return max(size, 0)
}

// download is one file being fetched.
type download struct {
	*clients
	src    string   // the resolve address, at the commit Hugging Face named
	file   *os.File // name.part
	record string   // name.part.json
	size   int64
	etag   string
	piece  int64
	pieces []piece // nil when the file is fetched in one request
	kept   int64

	mu    sync.Mutex
	taken int // pieces handed to a connection so far

	addrMu sync.Mutex
	url    string // where the bytes are served from; the CDN signs it for an hour
	gen    int    // how many times url has been asked for again

	checked atomic.Int64
	waiting atomic.Pointer[string] // why the last request failed, until bytes arrive again
}

// piece is a stretch of the file that one request asks for.
type piece struct {
	start, end int64
	done       atomic.Int64 // bytes from start that are on disk
}

func (p *piece) full() bool { return p.done.Load() == p.end-p.start }

// record is what name.part.json holds. The store reads its size and done to
// show an unfinished pull.
type record struct {
	Size  int64   `json:"size"`
	ETag  string  `json:"etag"`
	Piece int64   `json:"piece"`
	Done  []int64 `json:"done"`
}

// open takes name.part for this download and locks it against a second pull
// at once. A pull that finishes renames its name.part while still holding it,
// so the file locked here may have just become a finished model: it is taken
// only while the name still leads to it.
func (d *download) open(path string) error {
	for range 3 {
		f, err := openPart(path)
		if err != nil {
			return err
		}
		if err := lock(f); err != nil {
			f.Close()
			break
		}
		held, err := f.Stat()
		if err != nil {
			f.Close()
			return err
		}
		if named, err := os.Stat(path); err == nil && os.SameFile(held, named) {
			d.file, d.record = f, path+".json"
			return nil
		}
		f.Close()
	}
	return errors.New("another kinfer is pulling it already")
}

// prepare goes on from what an earlier pull of the same file left in
// name.part, or readies name.part to be filled from the start.
func (d *download) prepare() error {
	if d.size == 0 {
		return nil
	}
	if rec, ok := d.earlier(); ok {
		d.cut(rec.Piece)
		for i, done := range rec.Done {
			d.pieces[i].done.Store(done)
			d.kept += done
		}
		return nil
	}
	// Whatever the file holds is of another version, or on no record.
	d.cut(pieceSize)
	if err := d.file.Truncate(0); err != nil {
		return err
	}
	// Sparse: each piece takes disk space as it arrives.
	if err := d.file.Truncate(d.size); err != nil {
		return err
	}
	return d.save()
}

// earlier reads what an earlier pull of the same content left on record.
func (d *download) earlier() (record, bool) {
	var rec record
	b, err := os.ReadFile(d.record)
	if err != nil || json.Unmarshal(b, &rec) != nil {
		return rec, false
	}
	fi, err := d.file.Stat()
	if err != nil || fi.Size() != d.size || rec.Size != d.size || d.etag == "" || rec.ETag != d.etag || rec.Piece <= 0 {
		return rec, false
	}
	if int64(len(rec.Done)) != (d.size+rec.Piece-1)/rec.Piece {
		return rec, false
	}
	for i, done := range rec.Done {
		if done < 0 || done > min(rec.Piece, d.size-int64(i)*rec.Piece) {
			return rec, false
		}
	}
	return rec, true
}

func (d *download) cut(size int64) {
	d.piece = size
	d.pieces = make([]piece, (d.size+size-1)/size)
	for i := range d.pieces {
		d.pieces[i].start = int64(i) * size
		d.pieces[i].end = min(d.pieces[i].start+size, d.size)
	}
}

// save writes down how far each piece has got. The file goes to disk first,
// so the record is never ahead of it.
func (d *download) save() error {
	rec := record{Size: d.size, ETag: d.etag, Piece: d.piece, Done: make([]int64, len(d.pieces))}
	for i := range d.pieces {
		rec.Done[i] = d.pieces[i].done.Load()
	}
	if err := flush(d.file); err != nil {
		return err
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	tmp := d.record + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, d.record)
}

// forget removes the record of what has arrived.
func (d *download) forget() {
	os.Remove(d.record)
	os.Remove(d.record + ".tmp")
}

// finish gives the file its name: on disk whole first, so that no power cut
// can leave the name on a file the disk holds only part of, and while still
// locked, so that a pull starting this moment cannot take it for a name.part
// of its own.
func (d *download) finish(final string) error {
	if err := d.file.Sync(); err != nil {
		return err
	}
	if err := os.Rename(d.file.Name(), final); err != nil {
		return err
	}
	d.file.Close()
	d.forget()
	return nil
}

func (d *download) status() Status {
	var done int64
	for i := range d.pieces {
		done += d.pieces[i].done.Load()
	}
	st := Status{Done: done, Total: d.size, Kept: d.kept, Checked: d.checked.Load()}
	if why := d.waiting.Load(); why != nil {
		st.Waiting = *why
	}
	return st
}

// run fetches every piece still missing, conns at a time, and hashes the file
// from its start as it fills.
func (d *download) run(ctx context.Context, conns int, sum string, onProgress Progress) error {
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)

	var fetching sync.WaitGroup
	for range conns {
		fetching.Go(func() {
			buf := make([]byte, 1<<20)
			for p := d.take(); p != nil; p = d.take() {
				if err := d.fill(ctx, p, buf); err != nil {
					cancel(err)
					return
				}
			}
		})
	}
	fetched := make(chan struct{})
	go func() { fetching.Wait(); close(fetched) }()
	checked := make(chan error, 1)
	go func() { checked <- d.check(ctx, sum) }()

	tick := time.NewTicker(200 * time.Millisecond)
	defer tick.Stop()
	saved := time.Now()
	for fetched != nil || checked != nil {
		select {
		case <-fetched:
			fetched = nil
		case err := <-checked:
			checked = nil
			if err != nil {
				cancel(err)
			}
		case <-tick.C:
			st := d.status()
			onProgress(st)
			if st.Done < st.Total {
				if err := d.room(); err != nil {
					cancel(err)
				}
			}
			if time.Since(saved) >= saveEvery {
				if err := d.save(); err != nil {
					cancel(err)
				}
				saved = time.Now()
			}
		}
	}
	onProgress(d.status())

	err := context.Cause(ctx)
	if err != nil && !errors.Is(err, errMismatch) && !errors.Is(err, errNoRanges) {
		if serr := d.save(); serr != nil {
			err = fmt.Errorf("%w (and recording what arrived: %v)", err, serr)
		}
	}
	return err
}

// room stops a pull that has brought its disk down to diskFloor.
func (d *download) room() error {
	avail, err := diskFree(filepath.Dir(d.record))
	if err != nil || avail >= diskFloor {
		return nil
	}
	return fmt.Errorf("%w: %s free, and a pull leaves %s", errDiskFull, store.HumanSize(avail), store.HumanSize(diskFloor))
}

// take hands out the next piece with bytes still missing.
func (d *download) take() *piece {
	d.mu.Lock()
	defer d.mu.Unlock()
	for d.taken < len(d.pieces) {
		p := &d.pieces[d.taken]
		d.taken++
		if !p.full() {
			return p
		}
	}
	return nil
}

// fill fetches the rest of a piece, asking again from where it got to when a
// request fails, until patience has gone by without a byte.
func (d *download) fill(ctx context.Context, p *piece, buf []byte) error {
	since := time.Now() // when the piece last got bytes
	for tries := 0; !p.full(); {
		before := p.done.Load()
		err := d.get(ctx, p, buf)
		if p.done.Load() > before {
			since, tries = time.Now(), 0
		}
		if err == nil {
			continue
		}
		if ctx.Err() != nil {
			return context.Cause(ctx)
		}
		if errors.As(err, new(permanent)) {
			return err
		}
		if time.Since(since) >= patience {
			return fmt.Errorf("bytes %d-%d: %w", p.start+p.done.Load(), p.end-1, err)
		}
		why := reason(err)
		d.waiting.Store(&why)
		if tries++; tries == 1 && errors.Is(err, errExpired) {
			continue // a fresh address is waiting
		}
		select {
		case <-ctx.Done():
			return context.Cause(ctx)
		case <-time.After(retryBase << min(tries-1, 5)):
		}
	}
	return nil
}

// get makes one request for what a piece still lacks, and writes what arrives.
func (d *download) get(ctx context.Context, p *piece, buf []byte) error {
	addr, gen := d.address()
	off := p.start + p.done.Load()

	rctx, stop := context.WithCancel(ctx)
	defer stop()
	idle := time.AfterFunc(idleLimit, stop)
	defer idle.Stop()

	req, err := http.NewRequestWithContext(rctx, http.MethodGet, addr, nil)
	if err != nil {
		return permanent{err}
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", off, p.end-1))
	resp, err := d.fetcher.Do(req)
	if err != nil {
		return stalled(ctx, rctx, bare(err))
	}
	defer resp.Body.Close()

	switch code := resp.StatusCode; {
	case code == http.StatusPartialContent:
	case code == http.StatusOK:
		return permanent{errNoRanges}
	case code == http.StatusUnauthorized, code == http.StatusForbidden,
		code == http.StatusNotFound, code == http.StatusGone:
		// The CDN signs its address for an hour, and a big file takes longer.
		if err := d.refresh(ctx, gen); err != nil {
			return err
		}
		return fmt.Errorf("%w: %s", errExpired, resp.Status)
	case code == http.StatusTooManyRequests, code >= 500:
		return fmt.Errorf("the CDN answered %s", resp.Status)
	default:
		return permanent{fmt.Errorf("asked for bytes %d-%d: %s", off, p.end-1, resp.Status)}
	}
	if start, ok := rangeStart(resp.Header.Get("Content-Range")); !ok || start != off {
		return permanent{fmt.Errorf("asked for bytes from %d, got %q", off, resp.Header.Get("Content-Range"))}
	}

	body := watched{io.LimitReader(resp.Body, p.end-off), idle}
	for off < p.end {
		n, err := io.ReadFull(body, buf[:min(int64(len(buf)), p.end-off)])
		if n > 0 {
			if _, err := d.file.WriteAt(buf[:n], off); err != nil {
				return permanent{err}
			}
			off += int64(n)
			p.done.Add(int64(n))
			if d.waiting.Load() != nil {
				d.waiting.Store(nil)
			}
		}
		if err != nil {
			return stalled(ctx, rctx, err)
		}
	}
	return nil
}

// stalled says so when a request ended because nothing arrived, rather than
// passing on the cancellation that ended it.
func stalled(ctx, rctx context.Context, err error) error {
	if ctx.Err() == nil && rctx.Err() != nil {
		return fmt.Errorf("nothing arrived for %s", idleLimit)
	}
	return err
}

// bare drops the address from a failed request's error. The CDN's is a few
// hundred characters, most of them signature.
func bare(err error) error {
	var uerr *url.Error
	if errors.As(err, &uerr) {
		return uerr.Err
	}
	return err
}

// reason is what an error says, short enough for a progress line: without the
// address, and without the advice that comes after it.
func reason(err error) string {
	s, _, _ := strings.Cut(bare(err).Error(), "\n")
	return s
}

// watched puts off an idle timer each time bytes arrive through it.
type watched struct {
	r    io.Reader
	idle *time.Timer
}

func (w watched) Read(b []byte) (int, error) {
	n, err := w.r.Read(b)
	if n > 0 {
		w.idle.Reset(idleLimit)
	}
	return n, err
}

// address is where to ask for bytes, and how many times it has been renewed.
func (d *download) address() (string, int) {
	d.addrMu.Lock()
	defer d.addrMu.Unlock()
	return d.url, d.gen
}

// refresh asks Hugging Face for a new address, unless another connection
// already has since gen.
func (d *download) refresh(ctx context.Context, gen int) error {
	d.addrMu.Lock()
	defer d.addrMu.Unlock()
	if d.gen != gen {
		return nil
	}
	r, err := d.resolve(ctx, d.src)
	if err != nil {
		return err
	}
	if r.etag != d.etag || r.size != d.size {
		return permanent{errors.New("the file changed on Hugging Face during the pull")}
	}
	d.url, d.gen = r.url, d.gen+1
	return nil
}

// check hashes the file from its start as the pieces fill it, a few pieces
// behind the fastest connection, so the sum is ready moments after the last
// byte; after a resume it starts by reading back what the earlier pull
// fetched, which is what a power cut could have taken without the record
// knowing. It compares the sum with want.
func (d *download) check(ctx context.Context, want string) error {
	if want == "" {
		return nil
	}
	h := sha256.New()
	buf := make([]byte, 4<<20)
	for off := int64(0); off < d.size; {
		if ctx.Err() != nil {
			return context.Cause(ctx)
		}
		ready := d.ready()
		if ready == off {
			select {
			case <-ctx.Done():
				return context.Cause(ctx)
			case <-time.After(50 * time.Millisecond):
			}
			continue
		}
		n := min(int64(len(buf)), ready-off)
		if _, err := d.file.ReadAt(buf[:n], off); err != nil {
			return err
		}
		h.Write(buf[:n])
		off += n
		d.checked.Store(off)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return fmt.Errorf("%w (its sha256 is %s, not %s)", errMismatch, got, want)
	}
	return nil
}

// ready is how far from its start the file is on disk without a gap.
func (d *download) ready() int64 {
	for i := range d.pieces {
		p := &d.pieces[i]
		if done := p.done.Load(); done < p.end-p.start {
			return p.start + done
		}
	}
	return d.size
}

// stream fetches the file in one request from its start, for a server that
// does not say how long it is or ignores ranged requests. What it fetches
// cannot be gone on from, so nothing of it is kept when it stops.
func (d *download) stream(ctx context.Context, sum string, onProgress Progress) error {
	d.pieces = nil
	d.forget()
	if err := d.file.Truncate(0); err != nil {
		return err
	}

	rctx, stop := context.WithCancel(ctx)
	defer stop()
	idle := time.AfterFunc(idleLimit, stop)
	defer idle.Stop()

	addr, _ := d.address()
	req, err := http.NewRequestWithContext(rctx, http.MethodGet, addr, nil)
	if err != nil {
		return err
	}
	resp, err := d.fetcher.Do(req)
	if err != nil {
		return stalled(ctx, rctx, bare(err))
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return errors.New(resp.Status)
	}
	if d.size == 0 {
		d.size = max(resp.ContentLength, 0)
	}

	h := sha256.New()
	n := &counter{total: d.size, onProgress: onProgress, room: d.room}
	if _, err := io.Copy(io.MultiWriter(d.file, h, n), watched{resp.Body, idle}); err != nil {
		return stalled(ctx, rctx, err)
	}
	if sum != "" && hex.EncodeToString(h.Sum(nil)) != sum {
		return errMismatch
	}
	return nil
}

// counter reports the bytes written through it, at most five times a second,
// and stops them when the disk is nearly full.
type counter struct {
	done, total int64
	reported    time.Time
	onProgress  Progress
	room        func() error
}

func (c *counter) Write(b []byte) (int, error) {
	c.done += int64(len(b))
	if time.Since(c.reported) > 200*time.Millisecond {
		if err := c.room(); err != nil {
			return 0, err
		}
		c.onProgress(Status{Done: c.done, Total: c.total})
		c.reported = time.Now()
	}
	return len(b), nil
}

// failed says why a pull stopped and what it left behind. What a ranged pull
// fetched is kept for the next one; anything else is removed.
func (d *download) failed(ctx context.Context, filename string, err error) error {
	st := d.status()
	if d.pieces == nil || st.Done == 0 || errors.Is(err, errMismatch) {
		os.Remove(d.file.Name())
		d.forget()
		switch {
		case errors.Is(err, errMismatch):
			return fmt.Errorf("%s: %w; it was deleted, and the next pull starts over", filename, err)
		case ctx.Err() != nil:
			return fmt.Errorf("%s: stopped", filename)
		}
		return fmt.Errorf("download %s: %w", filename, err)
	}
	kept := fmt.Sprintf("%s of %s is in %s; the same pull goes on from there",
		store.HumanSize(st.Done), store.HumanSize(st.Total), d.file.Name())
	if ctx.Err() != nil {
		return fmt.Errorf("%s: stopped. %s", filename, kept)
	}
	return fmt.Errorf("download %s: %w\n  %s", filename, err, kept)
}

// clients are the two ways a download talks HTTP.
type clients struct {
	resolver *http.Client // stops at the redirect that describes the file
	fetcher  *http.Client
}

func newClients(conns int) *clients {
	// A transport of its own, not a copy of http.DefaultTransport: once that
	// has been used (List uses it), its TLS settings offer HTTP/2, and a copy
	// goes on offering it to servers it then cannot understand.
	t := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 30 * time.Second,
		IdleConnTimeout:       90 * time.Second,
		MaxIdleConnsPerHost:   conns, // kept for the next piece rather than closed
	}
	// Over HTTP/2 every request to a host shares one TCP connection, and with
	// it one connection's speed: the very limit the pieces are fetched in
	// parallel to get past. Over HTTP/1.1 each has a connection of its own.
	t.Protocols = new(http.Protocols)
	t.Protocols.SetHTTP1(true)
	return &clients{
		resolver: &http.Client{Transport: t, CheckRedirect: toCDN},
		fetcher:  &http.Client{Transport: t},
	}
}

// toCDN follows redirects up to the one that describes the file, carrying
// X-Linked-Size, and stops there: its Location is the CDN's address. Before it
// may come a renamed repo's redirect, or a mirror's to huggingface.co, which is
// how hf-mirror.com answers a machine in the US.
func toCDN(req *http.Request, via []*http.Request) error {
	if h := req.Response.Header; h.Get("X-Linked-Size") != "" || h.Get("X-Linked-Etag") != "" {
		return http.ErrUseLastResponse
	}
	if len(via) >= 10 {
		return errRedirects
	}
	return nil
}

// remote is what Hugging Face says about a file before sending it.
type remote struct {
	url    string // where its bytes are served from
	size   int64  // 0 when not said
	etag   string // names its content
	sha256 string // hex; Hugging Face has one for every LFS file
	commit string // the commit main is at
}

var sha256Hex = regexp.MustCompile(`^[0-9a-f]{64}$`)

// resolveSoon is resolve, asked again while the network or Hugging Face is
// down, for up to patience: between the pieces of a split model, a pull is
// not watched any more closely than in the middle of one.
func (c *clients) resolveSoon(ctx context.Context, src string, onProgress Progress) (remote, error) {
	start := time.Now()
	for tries := 0; ; tries++ {
		r, err := c.resolve(ctx, src)
		if err == nil || errors.As(err, new(permanent)) || ctx.Err() != nil || time.Since(start) >= patience {
			return r, err
		}
		onProgress(Status{Waiting: reason(err)})
		select {
		case <-ctx.Done():
			return remote{}, ctx.Err()
		case <-time.After(retryBase << min(tries, 5)):
		}
	}
}

// resolve asks Hugging Face about a file without fetching it. Its answer to a
// resolve address is a redirect to the CDN carrying the file's size, sha256
// and commit (X-Linked-Size, X-Linked-Etag, X-Repo-Commit). Only that header
// is a sha256: the CDN's own ETag is a hash of another kind.
func (c *clients) resolve(ctx context.Context, src string) (remote, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, src, nil)
	if err != nil {
		return remote{}, permanent{err}
	}
	resp, err := c.resolver.Do(req)
	if errors.Is(err, errRedirects) {
		return remote{}, permanent{err}
	}
	if err != nil {
		hint := ""
		if endpoint() == defaultEndpoint {
			hint = "\n  (unreachable? try: export HF_ENDPOINT=https://hf-mirror.com)"
		}
		return remote{}, fmt.Errorf("%w%s", err, hint)
	}
	resp.Body.Close()

	r := remote{etag: resp.Header.Get("X-Linked-Etag"), commit: resp.Header.Get("X-Repo-Commit")}
	if sum := strings.Trim(r.etag, `"`); sha256Hex.MatchString(sum) {
		r.sha256 = sum
	}
	if r.etag == "" {
		r.etag = resp.Header.Get("ETag")
	}
	r.size, _ = strconv.ParseInt(resp.Header.Get("X-Linked-Size"), 10, 64)

	switch code := resp.StatusCode; {
	case code >= 300 && code < 400:
		loc, err := resp.Location()
		if err != nil {
			return remote{}, permanent{err}
		}
		r.url = loc.String()
	case code == http.StatusOK:
		r.url = resp.Request.URL.String()
		if r.size == 0 {
			r.size = max(resp.ContentLength, 0)
		}
	case code == http.StatusUnauthorized, code == http.StatusForbidden:
		return remote{}, permanent{errors.New("not allowed: the repo is private or gated")}
	case code == http.StatusNotFound:
		return remote{}, permanent{errors.New("no such file")}
	case code == http.StatusTooManyRequests, code >= 500:
		return remote{}, fmt.Errorf("Hugging Face answered %s", resp.Status)
	default:
		return remote{}, permanent{errors.New(resp.Status)}
	}
	return r, nil
}

// rangeStart reads the first byte's offset from a Content-Range header, as in
// "bytes 100-199/491400032".
func rangeStart(h string) (int64, bool) {
	rest, ok := strings.CutPrefix(h, "bytes ")
	if !ok {
		return 0, false
	}
	first, _, ok := strings.Cut(rest, "-")
	if !ok {
		return 0, false
	}
	n, err := strconv.ParseInt(first, 10, 64)
	return n, err == nil
}
