// Command kinfer runs local models: pull them, list them, remove them, and
// serve them over HTTP to whatever wants to talk to a model.
package main

import (
	"context"
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
  kinfer list                  show installed models
  kinfer rm <model>            delete a model
  kinfer run <model> [prompt]  generate once from the command line
  kinfer serve                 serve models over HTTP
  kinfer fit                   what this machine can actually run

Examples:
  kinfer pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M
  kinfer run qwen "explain merkle trees"
  kinfer serve -addr :11500
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
	case "serve":
		err = cmdServe(os.Args[2:])
	case "fit":
		err = cmdFit(os.Args[2:])
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
	st, err := store.Open()
	if err != nil {
		return err
	}
	models, err := st.List()
	if err != nil {
		return err
	}
	if len(models) == 0 {
		fmt.Printf("no models in %s\n\ntry: kinfer pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M\n", st.Root())
		return nil
	}

	fmt.Printf("%-46s %10s  %s\n", "NAME", "SIZE", "MODIFIED")
	for _, m := range models {
		fmt.Printf("%-46s %10s  %s\n", m.Name, store.HumanSize(m.Size), humanTime(m.Modified))
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
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		return fmt.Errorf("usage: kinfer run <model> [prompt]")
	}

	st, err := store.Open()
	if err != nil {
		return err
	}
	path, err := st.Resolve(fs.Arg(0))
	if err != nil {
		return err
	}

	eng, err := engine.Open(path, engine.Options{GPULayers: *ngl, ContextSize: *nCtx})
	if err != nil {
		return err
	}
	defer eng.Close()

	prompt := strings.Join(fs.Args()[1:], " ")
	if prompt == "" {
		return fmt.Errorf("no prompt given")
	}

	var msgs []chat.Message
	if *system != "" {
		msgs = append(msgs, chat.Message{Role: "system", Content: *system})
	}
	msgs = append(msgs, chat.Message{Role: "user", Content: prompt})

	params := engine.DefaultGenParams()
	params.MaxTokens = *maxTok

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	_, err = eng.Chat(ctx, msgs, params, func(frag string) { fmt.Print(frag) })
	fmt.Println()
	return err
}

func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", ":11500", "listen address")
	ngl := fs.Int("ngl", 99, "layers to offload to GPU (0 = CPU only)")
	nCtx := fs.Int("ctx", 4096, "context size in tokens")
	if err := fs.Parse(args); err != nil {
		return err
	}

	st, err := store.Open()
	if err != nil {
		return err
	}
	srv := server.New(st, engine.Options{GPULayers: *ngl, ContextSize: *nCtx})
	defer srv.Close()

	models, _ := st.List()
	fmt.Printf("kinfer serving on %s — %d model(s) in %s\n", *addr, len(models), st.Root())
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
