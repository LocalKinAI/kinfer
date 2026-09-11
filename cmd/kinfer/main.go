// Command kinfer runs local models: pull them, list them, remove them, and
// serve them over HTTP to whatever wants to talk to a model.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/hardware"
	"github.com/LocalKinAI/kinfer/internal/hub"
	"github.com/LocalKinAI/kinfer/internal/server"
	"github.com/LocalKinAI/kinfer/internal/store"
)

const usage = `kinfer — a single-file local inference runtime

  kinfer pull <repo>[:quant]   download a model from Hugging Face
  kinfer list                  show models kinfer can load, including Ollama's
  kinfer rm <model>            delete a model
  kinfer run <model> [prompt]  chat with a model (no prompt = interactive)
  kinfer ps                    show which model is loaded right now
  kinfer plan <model>          what -slots and -ctx to serve it with
  kinfer serve                 serve models over HTTP
  kinfer fit                   what this machine can actually run
  kinfer can-run <hf-repo>     whether a model on Hugging Face will run here

Examples:
  kinfer pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M
  kinfer run qwen                        # interactive, model stays warm
  kinfer run qwen "explain merkle trees" # one shot, same warm model
  kinfer serve -addr :11500

run starts a background daemon on first use and reuses it afterwards, so the
model is loaded once rather than once per command. kinfer ps shows what it is
holding; -local skips the daemon and loads in-process instead.
`

func main() {
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(2)
	}

	var err error
	switch os.Args[1] {
	case "pull":
		err = cmdPull(os.Args[2:])
	case "list", "ls":
		err = cmdList(os.Args[2:])
	case "rm", "remove":
		err = cmdRemove(os.Args[2:])
	case "run":
		err = cmdRun(os.Args[2:])
	case "ps":
		err = cmdPS(os.Args[2:])
	case "serve":
		err = cmdServe(os.Args[2:])
	case "fit":
		err = cmdFit(os.Args[2:])
	case "plan":
		err = cmdPlan(os.Args[2:])
	case "can-run":
		err = cmdCanRun(os.Args[2:])
	case "-h", "--help", "help":
		fmt.Print(usage)
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "kinfer: %v\n", err)
		os.Exit(1)
	}
}

func cmdPull(args []string) error {
	fs := flag.NewFlagSet("pull", flag.ExitOnError)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		return fmt.Errorf("usage: kinfer pull <repo>[:quant]")
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

	fmt.Printf("querying %s via %s…\n", ref.Repo, hub.Endpoint())
	files, err := hub.List(ctx, ref.Repo)
	if err != nil {
		return err
	}
	pick, err := hub.Pick(files, ref)
	if err != nil {
		return err
	}

	// Hugging Face's model API omits file sizes unless asked for blobs, so
	// pick.Size is usually 0. The real size arrives with the download headers.
	if pick.Size > 0 {
		fmt.Printf("pulling %s (%s)\n", pick.Name, store.HumanSize(pick.Size))
	} else {
		fmt.Printf("pulling %s\n", pick.Name)
	}
	path, err := hub.Download(ctx, ref.Repo, pick.Name, st.Root(), func(done, total int64) {
		if total > 0 {
			fmt.Printf("\r  %s / %s  (%.0f%%)   ",
				store.HumanSize(done), store.HumanSize(total), float64(done)/float64(total)*100)
		} else {
			fmt.Printf("\r  %s   ", store.HumanSize(done))
		}
	})
	fmt.Println()
	if err != nil {
		return err
	}

	fmt.Printf("✅ %s\n", path)
	return nil
}

func cmdList(args []string) error {
	fs := flag.NewFlagSet("list", flag.ExitOnError)
	own := fs.Bool("own", false, "only models in kinfer's own directory")
	if err := fs.Parse(args); err != nil {
		return err
	}

	st, err := store.Open()
	if err != nil {
		return err
	}

	list := st.All
	if *own {
		list = st.List
	}
	models, err := list()
	if err != nil {
		return err
	}
	if len(models) == 0 {
		fmt.Println("no models yet — kinfer pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M")
		return nil
	}

	fmt.Printf("%-44s %10s  %-12s %s\n", "NAME", "SIZE", "MODIFIED", "SOURCE")
	for _, m := range models {
		size, when := store.HumanSize(m.Size), humanTime(m.Modified)
		if m.Remote() {
			// There is no file, so there is no size and no mtime. Printing the
			// zero values gives "0 B" and an age counted from year one, which
			// reads as a corrupt local model rather than a hosted one.
			size, when = "—", "—"
		}
		fmt.Printf("%-44s %10s  %-12s %s\n", m.Name, size, when, m.Source)
	}
	return nil
}

func cmdRemove(args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: kinfer rm <model>")
	}
	st, err := store.Open()
	if err != nil {
		return err
	}
	for _, name := range args {
		if err := st.Remove(name); err != nil {
			return err
		}
		fmt.Printf("removed %s\n", name)
	}
	return nil
}

func cmdRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	ngl := fs.Int("ngl", 99, "layers to offload to GPU (0 = CPU only)")
	nCtx := fs.Int("ctx", 4096, "context size in tokens")
	system := fs.String("system", "", "system prompt")
	maxTok := fs.Int("n", 512, "maximum tokens to generate")
	addr := fs.String("addr", defaultAddr, "daemon address")
	keepAlive := fs.Duration("keepalive", 5*time.Minute, "how long the daemon keeps an idle model loaded")
	local := fs.Bool("local", false, "load the model in this process instead of using a daemon")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		return fmt.Errorf("usage: kinfer run <model> [prompt]")
	}

	model := fs.Arg(0)
	rest := fs.Args()[1:]

	// Everything after the model name is the prompt, verbatim — a prompt is
	// allowed to contain anything, including text that looks like a flag. The
	// cost is that trailing flags are NOT parsed: Go's flag package stops at
	// the first positional argument, so `run qwen "hi" -n 20` would quietly ask
	// the model about "-n 20". Refuse that instead of asking a question nobody
	// typed. Quoted prompts are unaffected: "what does -n mean" is one argument
	// and never equals "-n".
	if bad := strayFlag(fs, rest); bad != "" {
		return fmt.Errorf("%s looks like a flag but comes after the model name, "+
			"so it would be sent to the model as text\n"+
			"       put flags first: kinfer run %s <model> [prompt]", bad, bad)
	}
	prompt := strings.Join(rest, " ")

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if *local {
		return runLocal(ctx, model, prompt, *system, *ngl, *nCtx, *maxTok)
	}

	base, err := ensureDaemon(*addr, *ngl, *nCtx, *keepAlive)
	if err != nil {
		return err
	}

	if prompt == "" {
		return repl(ctx, base, model, *system, *maxTok)
	}

	var msgs []clientMsg
	if *system != "" {
		msgs = append(msgs, clientMsg{Role: "system", Content: *system})
	}
	msgs = append(msgs, clientMsg{Role: "user", Content: prompt})

	_, err = streamChat(ctx, base, model, msgs, *maxTok, os.Stdout)
	fmt.Println()
	return err
}

// strayFlag returns the first argument that names a flag this command defines.
func strayFlag(fs *flag.FlagSet, args []string) string {
	defined := map[string]bool{}
	fs.VisitAll(func(f *flag.Flag) { defined[f.Name] = true })
	for _, a := range args {
		name := strings.TrimLeft(a, "-")
		if name == a || name == "" {
			continue // not flag-shaped
		}
		if i := strings.IndexByte(name, '='); i >= 0 {
			name = name[:i]
		}
		if defined[name] {
			return a
		}
	}
	return ""
}

// runLocal loads the model in this process — no daemon, nothing left running.
//
// Kept because it is the only way to get llama.cpp's own stderr in front of
// you, which is what you want when the question is "why did this model fail to
// load" rather than "what does this model say".
func runLocal(ctx context.Context, model, prompt, system string, ngl, nCtx, maxTok int) error {
	if prompt == "" {
		return fmt.Errorf("-local needs a prompt (interactive mode requires the daemon)")
	}

	st, err := store.Open()
	if err != nil {
		return err
	}
	path, err := st.Resolve(model)
	if err != nil {
		return err
	}

	eng, err := engine.Open(path, engine.Options{GPULayers: ngl, ContextSize: nCtx})
	if err != nil {
		return err
	}
	defer eng.Close()

	var msgs []chat.Message
	if system != "" {
		msgs = append(msgs, chat.Message{Role: "system", Content: system})
	}
	msgs = append(msgs, chat.Message{Role: "user", Content: prompt})

	params := engine.DefaultGenParams()
	params.MaxTokens = maxTok

	_, err = eng.Chat(ctx, msgs, params, func(frag string) { fmt.Print(frag) })
	fmt.Println()
	return err
}

// cmdPS reports what the daemon currently holds. It never starts one: "nothing
// is loaded" and "no daemon is running" are different answers, and conflating
// them would hide the case where a daemon died.
func cmdPS(args []string) error {
	fs := flag.NewFlagSet("ps", flag.ExitOnError)
	addr := fs.String("addr", defaultAddr, "daemon address")
	if err := fs.Parse(args); err != nil {
		return err
	}

	base := daemonURL(*addr)
	if !daemonAlive(base) {
		fmt.Printf("no kinfer daemon on %s\n", *addr)
		return nil
	}

	resp, err := http.Get(base + "/api/ps")
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	var out struct {
		Models []struct {
			Name      string     `json:"name"`
			Size      int64      `json:"size"`
			ExpiresAt *time.Time `json:"expires_at"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return err
	}
	if len(out.Models) == 0 {
		fmt.Printf("daemon on %s — no model loaded\n", *addr)
		return nil
	}

	fmt.Printf("%-44s %10s  %s\n", "NAME", "SIZE", "UNTIL")
	for _, m := range out.Models {
		until := "no expiry"
		if m.ExpiresAt != nil {
			if d := time.Until(*m.ExpiresAt); d > 0 {
				until = d.Round(time.Second).String()
			} else {
				until = "expiring"
			}
		}
		fmt.Printf("%-44s %10s  %s\n", m.Name, store.HumanSize(m.Size), until)
	}
	return nil
}

func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", ":11500", "listen address")
	ngl := fs.Int("ngl", 99, "layers to offload to GPU (0 = CPU only)")
	nCtx := fs.Int("ctx", 4096, "context size per conversation, in tokens")
	slots := fs.Int("slots", engine.DefaultSlots, "conversations served at once (use 8, 32, 64 or 128 — never 12-16)")
	prefix := fs.Int("prefix", engine.DefaultPrefixSlots, "prompt prefixes kept resident so repeat requests skip prefilling them (-1 disables)")
	queue := fs.Int("queue", engine.DefaultMaxQueue, "requests that may wait for a slot before the server answers 503")
	upstream := fs.String("upstream", server.Upstream(),
		"where cloud models are served from (default: $OLLAMA_HOST, else Ollama's own address)")
	maxGen := fs.Duration("max-gen", engine.DefaultMaxGenerate, "wall-clock limit on one reply (0 removes the limit)")
	maxWait := fs.Duration("max-wait", engine.DefaultMaxWait, "how long a request may queue before being refused (0 removes the limit)")
	keepAlive := fs.Duration("keepalive", 5*time.Minute, "unload an idle model after this long (0 = never)")
	if err := fs.Parse(args); err != nil {
		return err
	}

	st, err := store.Open()
	if err != nil {
		return err
	}
	srv := server.New(st, engine.Options{
		GPULayers: *ngl, ContextSize: *nCtx, Slots: *slots, PrefixSlots: *prefix,
		MaxQueue: *queue, MaxGenerate: noLimit(*maxGen), MaxWait: noLimit(*maxWait),
	})
	srv.SetKeepAlive(*keepAlive)
	defer srv.Close()

	models, _ := st.List()
	fmt.Printf("kinfer serving on %s — %d model(s) in %s\n", *addr, len(models), st.Root())
	fmt.Printf("  %d slots × %d tokens — conversations share one forward pass\n", *slots, *nCtx)
	if *prefix > 0 {
		fmt.Printf("  %d prefix slots — a repeated system prompt is prefilled once\n", *prefix)
	}
	fmt.Printf("  Ollama API : POST %s/api/chat        GET %s/api/tags\n", *addr, *addr)
	fmt.Printf("  OpenAI API : POST %s/v1/chat/completions\n\n", *addr)
	fmt.Printf("  point LocalKin at it by setting a soul's brain.endpoint to %s\n\n", *addr)

	httpSrv := &http.Server{
		Addr:    *addr,
		Handler: srv.Handler(),
		// No write timeout: generation can legitimately run for minutes, and a
		// deadline here would sever a reply mid-sentence.
		ReadHeaderTimeout: 10 * time.Second,
	}

	srv.SetUpstream(*upstream)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		fmt.Println("\nshutting down…")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return httpSrv.Shutdown(shutdownCtx)
	}
}

// cmdFit reports what this machine can run, and marks which installed models
// are within reach.
func cmdFit(args []string) error {
	info := hardware.Detect()
	rec := hardware.Recommend(info)

	fmt.Printf("Machine\n  %s\n\n", info)

	if rec.Best == nil {
		fmt.Println("Could not size a recommendation for this machine.")
	} else {
		fmt.Printf("Recommended\n")
		fmt.Printf("  %s at %s  (~%s, %s)\n",
			rec.Best.Label(), rec.Best.Quant, store.HumanSize(rec.Best.EstBytes), rec.Best.Comfort)
		fmt.Printf("  kinfer pull %s\n\n", rec.Best.Example)

		if len(rec.Options) > 1 {
			fmt.Printf("Also fits\n")
			for _, o := range rec.Options {
				if rec.Best != nil && o.Params == rec.Best.Params {
					continue
				}
				fmt.Printf("  %-6s %9s  %s\n", o.Label(), store.HumanSize(o.EstBytes), o.Comfort)
			}
			fmt.Println()
		}
	}

	fmt.Printf("Backend\n  %s — %s\n\n", rec.Backend, rec.BackendNote)

	// Judge what is already installed against the same budget.
	if st, err := store.Open(); err == nil {
		if models, err := st.List(); err == nil && len(models) > 0 {
			fmt.Printf("Installed\n")
			for _, m := range models {
				fmt.Printf("  %-44s %9s  %s\n", m.Name, store.HumanSize(m.Size), rec.Fits(m.Size))
			}
			fmt.Println()
		}
	}

	if len(rec.Notes) > 0 {
		fmt.Printf("Notes\n")
		for _, n := range rec.Notes {
			fmt.Printf("  · %s\n", n)
		}
	}
	return nil
}

func humanTime(t time.Time) string {
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%d minutes ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%d hours ago", int(d.Hours()))
	default:
		return fmt.Sprintf("%d days ago", int(d.Hours()/24))
	}
}

// noLimit translates the command line's way of removing a limit into the
// package's.
//
// They differ because they answer different questions. engine.Options is a Go
// struct whose zero value has to mean "I did not set this", so a limit is
// removed there with a negative. A flag always has a value — its default is
// already the default — so on the command line 0 can mean what an operator
// expects it to mean, and "-max-gen 0" is how you say no limit.
//
// Writing "-max-gen -1" would not have worked in any case: Go parses a duration
// and "-1" has no unit, so the flag package prints an error and exits. Which it
// did, into a log nobody was reading, and cost an afternoon's confusion over a
// server that would not start.
func noLimit(d time.Duration) time.Duration {
	if d == 0 {
		return -1
	}
	return d
}
