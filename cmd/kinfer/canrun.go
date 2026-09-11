package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/hardware"
	"github.com/LocalKinAI/kinfer/internal/hub"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
	"github.com/LocalKinAI/kinfer/internal/store"
)

// cmdCanRun answers, without downloading anything, whether a model on Hugging
// Face will run here.
//
// Three things have to be true, and they are easy to confuse. The architecture
// must be one this build of llama.cpp implements; a GGUF conversion must exist,
// which only happens after the architecture is supported; and the smallest
// quantisation must fit in memory. The third is where most large models fail,
// and it is about total parameters — a mixture-of-experts model activates a
// fraction of its weights per token, which makes it fast, but every expert
// still has to be resident because routing may reach any of them.
func cmdCanRun(args []string) error {
	fs := flag.NewFlagSet("can-run", flag.ExitOnError)
	all := fs.Bool("all", false, "list every quantisation, not just the ones that fit")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		return fmt.Errorf("usage: kinfer can-run <hf-repo>   e.g. unsloth/Qwen3.8-Flash-Next-GGUF")
	}
	repo := fs.Arg(0)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	rec := hardware.Recommend(hardware.Detect())
	libDir, err := nativelib.Prepare()
	if err != nil {
		return err
	}

	fmt.Printf("%s\n\n", repo)

	// ── what it is ───────────────────────────────────────────────────────────
	//
	// The repo listing comes first and its failure is fatal. Carrying on would
	// print "architecture unknown" and "nothing fits", which read as findings
	// about the model rather than as a network that did not answer.
	quants, err := hub.Quants(ctx, repo)
	if err != nil {
		return err
	}
	arch, archFrom := "", ""
	var archErr error

	if len(quants) > 0 {
		// A GGUF repo states its architecture in the file itself, and a ranged
		// request reads it without fetching the model.
		a, err := hub.Architecture(ctx, repo, quants[0].Files[0])
		if err == nil {
			arch, archFrom = a, "GGUF metadata"
		} else {
			archErr = err
		}
	}
	if arch == "" {
		if a, err := hub.ModelType(ctx, repo); err == nil && a != "" {
			arch, archFrom = a, "config.json"
			archErr = nil
		}
	}

	switch {
	case arch == "" && archErr != nil:
		// "no metadata" and "the metadata could not be fetched" are different
		// claims, and only one of them is about the model.
		fmt.Printf("  architecture   could not be read: %v\n", archErr)

	case arch == "":
		fmt.Printf("  architecture   unknown — no GGUF metadata and no config.json\n")

	case archFrom == "config.json":
		// A repo's config.json names the architecture in Hugging Face's
		// vocabulary, and the GGUF converter renames it: DeepSeek-V4.1 is
		// "deepseek_v41" there and "deepseek4" in a GGUF. Checking one against
		// the other reports models as unrunnable that run perfectly well, so it
		// is not checked at all — only the GGUF's own name is authoritative.
		fmt.Printf("  architecture   %-16s (declared by config.json; a GGUF conversion renames it,\n", arch)
		fmt.Printf("                 %-16s  so support is decided from the converted file)\n", "")

	default:
		supported, known := llama.Supports(libDir, arch)
		mark := "?"
		note := "this build could not be inspected"
		if known {
			mark, note = "✗", "this build of llama.cpp does not implement it"
			if supported {
				mark, note = "✓", "implemented by the embedded llama.cpp"
			}
		}
		fmt.Printf("  architecture   %-16s %s %s  (from %s)\n", arch, mark, note, archFrom)
		if known && !supported {
			fmt.Printf("\n  Nothing else matters until that changes: no quantisation of an\n")
			fmt.Printf("  unimplemented architecture will load, whatever its size.\n")
			return nil
		}
	}

	// ── where the weights are ────────────────────────────────────────────────
	if len(quants) == 0 {
		fmt.Printf("  weights        safetensors only — kinfer needs GGUF\n\n")
		name := repo
		if i := strings.LastIndex(name, "/"); i >= 0 {
			name = name[i+1:]
		}
		hits, err := hub.Search(ctx, name+" GGUF", 8)
		if err != nil || len(hits) == 0 {
			fmt.Printf("  No GGUF conversion found. Someone has to convert it first.\n")
			return nil
		}
		fmt.Printf("  Community conversions to try:\n")
		for _, h := range hits {
			if !strings.EqualFold(h, repo) {
				fmt.Printf("    kinfer can-run %s\n", h)
			}
		}
		return nil
	}

	// ── what fits ────────────────────────────────────────────────────────────
	fmt.Printf("  machine        %s\n", hardware.Detect())
	fmt.Printf("  budget         %s for weights\n\n", store.HumanSize(rec.Budget))

	var fits, toobig []hub.Quant
	for _, q := range quants {
		if q.Size <= rec.Budget {
			fits = append(fits, q)
		} else {
			toobig = append(toobig, q)
		}
	}

	if len(fits) == 0 {
		fmt.Printf("  Nothing fits. The smallest is %s at %s.\n",
			quants[0].Name, store.HumanSize(quants[0].Size))
		fmt.Printf("  A mixture-of-experts model is no help here: it activates a fraction\n")
		fmt.Printf("  of its weights per token, which makes it fast, but routing can reach\n")
		fmt.Printf("  any expert, so all of them stay resident.\n")
		return nil
	}

	fmt.Printf("  %-22s %10s  %s\n", "QUANTISATION", "SIZE", "")
	for _, q := range fits {
		fmt.Printf("  %-22s %10s  %s\n", q.Name, store.HumanSize(q.Size), rec.Fits(q.Size))
	}
	if *all {
		for _, q := range toobig {
			fmt.Printf("  %-22s %10s  too large\n", q.Name, store.HumanSize(q.Size))
		}
	} else if len(toobig) > 0 {
		fmt.Printf("  (%d larger quantisations hidden — pass -all to see them)\n", len(toobig))
	}

	best := fits[len(fits)-1] // the largest that fits is the best quality
	fmt.Printf("\n  kinfer pull %s:%s\n", repo, best.Name)
	return nil
}
