package main

import (
	"flag"
	"fmt"

	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
	"github.com/LocalKinAI/kinfer/internal/store"
)

// kinfer plan says what -slots and -ctx this machine can actually serve a
// given model with.
//
// Distinct from `kinfer fit`, which answers the other direction — what size of
// model this machine could run at all. plan takes the model as given and sizes
// the cache around it.
//
// The loop it replaces: pick numbers, wait 34 seconds for 73 GiB to load, read
// the warning, pick again. Twice in one afternoon that loop was entered with
// numbers derived by scaling a cache cost linearly — which is wrong for a
// hybrid model, gave 1.7 GiB where the truth was 4.3, and left no headroom at
// all. This is that arithmetic written down where it cannot be mis-remembered.

func cmdPlan(args []string) error {
	fs := flag.NewFlagSet("plan", flag.ExitOnError)
	ctxWant := fs.Int("ctx", engine.DefaultContextSize, "context per conversation to plan for")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		return fmt.Errorf("usage: kinfer plan <model> [-ctx N]")
	}

	st, err := store.Open()
	if err != nil {
		return err
	}
	m, err := st.Lookup(fs.Arg(0))
	if err != nil {
		return err
	}
	if m.Remote() {
		fmt.Printf("%s is a cloud model — it is served elsewhere and occupies nothing here.\n", m.Name)
		return nil
	}

	budget, err := acceleratorBudget()
	if err != nil {
		return err
	}

	sh, shErr := store.ReadShape(m.Path)
	weights := m.Size
	free := budget - weights

	fmt.Printf("%s\n", m.Name)
	fmt.Printf("  weights        %10s\n", store.HumanSize(weights))
	fmt.Printf("  GPU budget     %10s   (this process's share, not the machine's RAM)\n",
		store.HumanSize(budget))
	fmt.Printf("  left for cache %10s\n\n", store.HumanSize(free))

	if free <= 0 {
		fmt.Printf("  This model does not fit. It is %s larger than the budget.\n",
			store.HumanSize(-free))
		return nil
	}
	if shErr != nil || sh.CacheBytes(1024) == 0 {
		fmt.Printf("  Cannot size the cache: the header does not carry the shape this needs.\n")
		fmt.Printf("  Load it once and read the memory line kinfer prints.\n")
		return nil
	}

	// Keep the same margin the server warns below, so fit does not recommend a
	// configuration the server will immediately complain about.
	spendable := free - int64(engine.ThinHeadroom())
	fmt.Printf("  %-8s %-14s %s\n", "-slots", "total tokens", "largest -ctx that fits")
	for _, slots := range []int{1, 2, 4, 8, 16, 32} {
		per := largestCtx(sh, spendable, slots)
		note := ""
		if per == 0 {
			note = "  (no room)"
		} else if sh.TrainedCtx > 0 && per > sh.TrainedCtx {
			per, note = sh.TrainedCtx, "  (capped at what it was trained for)"
		}
		fmt.Printf("  %-8d %-14d %d%s\n", slots, per*slots, per, note)
	}

	fmt.Printf("\n  For -ctx %d: ", *ctxWant)
	best := 0
	for _, slots := range []int{32, 16, 8, 4, 2, 1} {
		if largestCtx(sh, spendable, slots) >= *ctxWant {
			best = slots
			break
		}
	}
	if best == 0 {
		fmt.Printf("no slot count fits. Lower -ctx.\n")
	} else {
		fmt.Printf("kinfer serve -ctx %d -slots %d\n", *ctxWant, best)
	}

	if sh.Experts > 0 {
		fmt.Printf("\n  %s is a mixture of experts (%d experts). Concurrency still pays,\n"+
			"  but less than the slot count suggests: each token in a batch routes to\n"+
			"  its own experts, so they do not share a weight read. Measured 1.6x the\n"+
			"  step time for twice the tokens, not 1.0x.\n", m.Name, sh.Experts)
	}
	fmt.Printf("\n  An estimate. Hybrid architectures hold per-sequence state that does not\n" +
		"  scale with context, and cost more than this at high slot counts — the\n" +
		"  memory line printed at load time is the measurement.\n")
	return nil
}

// largestCtx is the biggest per-conversation context whose total cache fits,
// rounded down to a power of two so the answer is one an operator would pick.
func largestCtx(sh store.Shape, spendable int64, slots int) int {
	if spendable <= 0 {
		return 0
	}
	for ctx := 1 << 20; ctx >= 512; ctx /= 2 {
		if sh.CacheBytes(ctx*slots) <= spendable {
			return ctx
		}
	}
	return 0
}

func acceleratorBudget() (int64, error) {
	dir, err := nativelib.Prepare()
	if err != nil {
		return 0, err
	}
	if err := llama.Bind(dir); err != nil {
		return 0, err
	}
	d, ok := llama.Accelerator()
	if !ok {
		return 0, fmt.Errorf("no GPU found; kinfer fit has nothing to budget against")
	}
	return int64(d.Total), nil
}
