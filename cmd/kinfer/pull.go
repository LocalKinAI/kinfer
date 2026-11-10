package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/LocalKinAI/kinfer/internal/hub"
	"github.com/LocalKinAI/kinfer/internal/mlxasr"
	"github.com/LocalKinAI/kinfer/internal/mlxengine"
	"github.com/LocalKinAI/kinfer/internal/mlxmodel"
	"github.com/LocalKinAI/kinfer/internal/mlxtts"
	"github.com/LocalKinAI/kinfer/internal/store"
)

func cmdPull(args []string) error {
	fs := flag.NewFlagSet("pull", flag.ExitOnError)
	conns := fs.Int("connections", 8, "connections to fetch a file over at once")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		return fmt.Errorf("usage: kinfer pull [-connections n] <repo>[:quant]   (a GGUF repo, or an MLX one of safetensors)")
	}

	ref, err := hub.ParseRef(fs.Arg(0))
	if err != nil {
		return err
	}
	st, err := store.Open()
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	// The first Ctrl-C stops the pull and keeps what has arrived; a second,
	// should that take a moment, ends kinfer at once.
	context.AfterFunc(ctx, stop)

	fmt.Printf("querying %s via %s…\n", ref.Repo, hub.Endpoint())
	all, err := hub.Files(ctx, ref.Repo)
	if err != nil {
		return err
	}
	files := hub.GGUFs(all)
	if len(files) == 0 {
		if set, ok := hub.MLXFiles(all); ok {
			return pullMLX(ctx, st, ref, set, *conns)
		}
		return fmt.Errorf("%s contains no .gguf files, nor safetensors kinfer can serve on MLX", ref.Repo)
	}
	pick, err := hub.Pick(files, ref)
	if err != nil {
		return err
	}

	// A split model is no model without every piece.
	set := hub.Shards(files, pick)
	m := &pullMeter{tty: isTerminal(int(os.Stdout.Fd()))}
	if m.tty {
		m.width = termWidth(int(os.Stdout.Fd()))
	}
	for _, f := range set {
		m.total += f.Size
	}
	switch {
	case len(set) > 1:
		fmt.Printf("pulling %s and %d more pieces (%s)\n", set[0].Name, len(set)-1, store.HumanSize(m.total))
	case m.total > 0:
		fmt.Printf("pulling %s (%s)\n", pick.Name, store.HumanSize(m.total))
	default:
		fmt.Printf("pulling %s\n", pick.Name)
	}
	// A vision model reads pictures through its projector, which comes with
	// it, kept as NAME.mmproj.gguf beside it: every repo calls its own
	// mmproj-F16.gguf.
	want := set
	proj, sees := hub.Projector(files)
	if sees {
		m.total += proj.Size
		want = append(append([]hub.File(nil), set...), hub.File{Name: hub.ProjectorName(set), Size: proj.Size})
		fmt.Printf("  and %s (%s), the projector it reads pictures with\n", proj.Name, store.HumanSize(proj.Size))
	}
	if err := hub.Room(st.Root(), want); err != nil {
		return err
	}

	var first string
	checked := true
	for _, f := range set {
		m.start(f.Name)
		path, err := hub.Download(ctx, ref.Repo, f.Name, st.Root(), *conns, m.show)
		m.end()
		if err != nil {
			return err
		}
		if first == "" {
			first = path
		}
		checked = checked && m.last.Total > 0 && m.last.Checked == m.last.Total
	}
	if sees {
		m.start(proj.Name)
		_, err := hub.DownloadAs(ctx, ref.Repo, proj.Name, filepath.Join(st.Root(), hub.ProjectorName(set)), *conns, m.show)
		m.end()
		if err != nil {
			return err
		}
		checked = checked && m.last.Total > 0 && m.last.Checked == m.last.Total
	}
	if checked {
		fmt.Printf("✅ %s (sha256 matches Hugging Face's)\n", first)
	} else {
		fmt.Printf("✅ %s\n", first)
	}
	return nil
}

// servedOnMLX is a model_type kinfer serves on MLX: a chat model's
// (mlxmodel), a speech recognition model's (mlxasr) or a text to speech
// model's (mlxtts).
func servedOnMLX(arch string) bool {
	return mlxmodel.Serves(arch) || mlxasr.Serves(arch) || mlxtts.Serves(arch)
}

// mlxExperts checks that an MoE repo keeps its experts as the MLX engine
// reads them: caught before a pull, a layout it refuses costs a small JSON
// file instead of the whole download.
func mlxExperts(ctx context.Context, repo, arch string) error {
	if !strings.Contains(arch, "moe") {
		return nil
	}
	names, err := hub.TensorNames(ctx, repo)
	if err != nil {
		return err
	}
	return mlxmodel.ExpertsReadable(names)
}

// pullMLX fetches a model kinfer serves on MLX: the repo's safetensors and the
// files that describe them, into a directory of the store named for the repo.
func pullMLX(ctx context.Context, st *store.Store, ref hub.Ref, set []hub.File, conns int) error {
	if ref.Quant != "" || ref.File != "" {
		return fmt.Errorf("%s is safetensors, pulled whole: kinfer pull %s", ref.Repo, ref.Repo)
	}
	arch, err := hub.ModelType(ctx, ref.Repo)
	if err != nil {
		return err
	}
	if !servedOnMLX(arch) {
		return fmt.Errorf("%s is safetensors of model_type %q, which kinfer does not serve on MLX (Qwen3.5 and Qwen3.8, qwen3_5 and qwen3_5_moe, the speech models Qwen3-ASR and SenseVoice, qwen3_asr and sensevoice, and Qwen3-TTS CustomVoice, qwen3_tts, it does); a GGUF of it may run on llama.cpp: kinfer can-run %s",
			ref.Repo, arch, ref.Repo)
	}
	if err := mlxExperts(ctx, ref.Repo, arch); err != nil {
		return fmt.Errorf("%s: %w", ref.Repo, err)
	}
	if err := mlxengine.Usable(); err != nil {
		return err
	}

	name := path.Base(ref.Repo)
	dir := filepath.Join(st.Root(), name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	m := &pullMeter{tty: isTerminal(int(os.Stdout.Fd()))}
	if m.tty {
		m.width = termWidth(int(os.Stdout.Fd()))
	}
	for _, f := range set {
		m.total += f.Size
	}
	fmt.Printf("pulling %s, %d files (%s), served on MLX\n", name, len(set), store.HumanSize(m.total))
	if err := hub.Room(dir, set); err != nil {
		os.Remove(dir) // only if nothing is in it yet
		return err
	}

	// Only the weights are stored by sha256 on Hugging Face; the small files
	// are git's, with a hash of another kind.
	checked := true
	for _, f := range set {
		m.start(f.Name)
		// A file in a folder keeps it: Qwen3-TTS's speech_tokenizer/ holds a
		// model.safetensors of its own.
		dest := filepath.Join(dir, filepath.FromSlash(f.Name))
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			return err
		}
		_, err := hub.DownloadAs(ctx, ref.Repo, f.Name, dest, conns, m.show)
		m.end()
		if err != nil {
			return err
		}
		if strings.HasSuffix(f.Name, ".safetensors") {
			checked = checked && m.last.Total > 0 && m.last.Checked == m.last.Total
		}
	}
	if checked {
		fmt.Printf("✅ %s (the weights' sha256 matches Hugging Face's)\n", dir)
	} else {
		fmt.Printf("✅ %s\n", dir)
	}
	return nil
}

// pullMeter draws a pull's progress on one line: the bytes of every piece of
// the model, how fast this pull is fetching them, and how long the rest will
// take at that speed. Where no one watches a terminal — a log, or ssh without
// one — it prints the line every half minute instead of redrawing it.
type pullMeter struct {
	total   int64 // bytes in every file of the pull
	before  int64 // bytes in the files finished
	fetched int64 // what this pull fetched of those

	tty   bool // whether the line is redrawn in place
	width int  // the terminal's, which the line must not reach

	name    string // the file being fetched
	last    hub.Status
	told    bool      // whether what an earlier pull left of it has been said
	drawn   bool      // whether a progress line waits for its newline
	printed time.Time // when the line last went out, off a terminal
	recent  []sample  // what this pull had fetched, over the last few seconds
}

type sample struct {
	at  time.Time
	got int64
}

func (m *pullMeter) start(name string) {
	m.name, m.last, m.told = name, hub.Status{}, false
	m.printed = time.Now() // off a terminal, the first line comes when there is something to say
}

func (m *pullMeter) end() {
	m.before += m.last.Total
	m.fetched += m.last.Done - m.last.Kept
	if m.drawn {
		fmt.Println()
		m.drawn = false
	}
}

func (m *pullMeter) show(st hub.Status) {
	prev := m.last
	m.last = st
	if !m.told && st.Total > 0 {
		m.told = true
		switch {
		case st.Kept == st.Total:
			m.say(fmt.Sprintf("  %s is here already", m.name))
		case st.Kept > 0:
			m.say(fmt.Sprintf("  %s: going on from %s of %s", m.name, store.HumanSize(st.Kept), store.HumanSize(st.Total)))
		}
	}
	if st.Total > 0 && st.Kept == st.Total {
		return
	}

	now := time.Now()
	got := m.fetched + st.Done - st.Kept
	m.recent = append(m.recent, sample{now, got})
	for len(m.recent) > 2 && now.Sub(m.recent[0].at) > 5*time.Second {
		m.recent = m.recent[1:]
	}

	done, total := m.before+st.Done, max(m.total, m.before+st.Total)
	line := "  " + store.HumanSize(done)
	if total > 0 {
		line += fmt.Sprintf(" / %s  (%.0f%%)", store.HumanSize(total), float64(done)/float64(total)*100)
	}
	switch span := now.Sub(m.recent[0].at); {
	case st.Waiting != "" && st.Done == prev.Done:
		// Nothing arrived since the last line: say why rather than 0 B/s.
		line += "  waiting: " + st.Waiting
	case span >= time.Second:
		rate := float64(got-m.recent[0].got) / span.Seconds()
		line += fmt.Sprintf("  %s/s", store.HumanSize(int64(rate)))
		if rate > 0 && total > done {
			left := time.Duration(float64(total-done) / rate * float64(time.Second))
			line += fmt.Sprintf("  %s left", left.Round(time.Second))
		}
	}
	if st.Total > 0 && st.Done == st.Total && st.Checked > 0 && st.Checked < st.Total {
		line = fmt.Sprintf("  checking sha256: %s / %s", store.HumanSize(st.Checked), store.HumanSize(st.Total))
	}
	m.draw(line)
}

// say prints a line of its own, under the progress line if one is drawn.
func (m *pullMeter) say(s string) {
	if m.drawn {
		fmt.Println()
		m.drawn = false
	}
	fmt.Println(s)
}

// draw redraws the progress line in place, cut to the terminal's width so it
// cannot wrap; off a terminal, it prints it every half minute.
func (m *pullMeter) draw(line string) {
	if !m.tty {
		if time.Since(m.printed) >= 30*time.Second {
			fmt.Println(line)
			m.printed = time.Now()
		}
		return
	}
	w := max(m.width-1, 20)
	if r := []rune(line); len(r) > w {
		line = string(r[:w])
	}
	fmt.Printf("\r%-*s", w, line)
	m.drawn = true
}
