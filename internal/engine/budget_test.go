package engine

import (
	"testing"

	"github.com/LocalKinAI/kinfer/internal/llama"
)

// The numbers below are the ones measured on the 96 GB Mac Studio that prompted
// this code: a 77.8 GiB Metal budget, a 73.4 GiB model, and two configurations
// either side of the line — -slots 2 (1.7 GiB of cache, served for hours) and
// -slots 4 (2.7 GiB, died mid-batch).
const (
	budgetTotal  = 79667 << 20 // 77.8 GiB, Metal's recommendedMaxWorkingSetSize
	modelCost    = 75161 << 20 // 73.4 GiB
	ctxCost2Slot = 1740 << 20  // 1.7 GiB  -> 2.7 GiB free
	ctxCost4Slot = 2765 << 20  // 2.7 GiB  -> 1.7 GiB free
)

func budgetFor(total, model, ctx uint64) *budget {
	return &budget{
		dev:        llama.Device{Name: "MTL0", Type: llama.DeviceGPU, Total: total},
		known:      true,
		atStart:    total,
		afterModel: total - model,
		afterCtx:   total - model - ctx,
	}
}

// The suggestion has to be smaller than what the operator asked for, or it is
// not advice — it is an echo.
func TestSuggestCtxShrinks(t *testing.T) {
	b := budgetFor(budgetTotal, modelCost, ctxCost4Slot)
	got := suggestCtx(b, 4, 0, 16384)
	if got >= 16384 {
		t.Errorf("suggestCtx = %d, which is no smaller than the 16384 that did not fit", got)
	}
	if got <= 0 {
		t.Errorf("suggestCtx = %d, which is not a context anyone can serve", got)
	}
	if got&(got-1) != 0 {
		t.Errorf("suggestCtx = %d, want a power of two so it reads like a setting", got)
	}
}

// Acting on the suggestion has to actually clear the line it was derived from.
// A suggestion that still warns would send the operator round the loop again.
func TestSuggestedContextClearsTheWarning(t *testing.T) {
	for _, slots := range []int{2, 4, 8} {
		b := budgetFor(budgetTotal, modelCost, ctxCost4Slot)
		got := suggestCtx(b, slots, 0, 16384)
		free := freeAt(b, got, slots, 0, 16384)
		if free < thinHeadroom {
			t.Errorf("slots=%d: suggested -ctx %d still leaves %s, under the %s line",
				slots, got, gib(free), gib(thinHeadroom))
		}
	}
}

// The notice must fire for the configuration that died and stay quiet for the
// one that served for hours. That is a smaller claim than it looks: the failure
// did not reproduce afterwards, so this pins the line where it was drawn rather
// than asserting the line predicts anything.
func TestThinHeadroomSeparatesTheMeasuredCases(t *testing.T) {
	thin := func(b *budget) bool { return b.afterCtx < thinHeadroom }
	worked := budgetFor(budgetTotal, modelCost, ctxCost2Slot) // 2.7 GiB free
	died := budgetFor(budgetTotal, modelCost, ctxCost4Slot)   // 1.6 GiB free

	if !thin(died) {
		t.Errorf("the configuration that died left %s and was not called thin", gib(died.afterCtx))
	}
	// And it must stay quiet for the one that served for hours. A warning that
	// fires on a working configuration is noise, which is exactly what the
	// first version of this file did at a 5% line.
	if thin(worked) {
		t.Errorf("the configuration that served for hours left %s and was called thin",
			gib(worked.afterCtx))
	}
}

// A machine with no accelerator has no budget to police, and must not be
// reported on as if it had one.
func TestBudgetWithoutAnAcceleratorSaysNothing(t *testing.T) {
	b := &budget{} // known == false
	b.report(4, 0, 16384)
	if b.sample() != 0 {
		t.Error("sampling a nonexistent device returned a number")
	}
}

func TestSubDoesNotWrap(t *testing.T) {
	if got := sub(1, 5); got != 0 {
		t.Errorf("sub(1,5) = %d, want 0 — an unsigned wrap here would print exabytes", got)
	}
}
