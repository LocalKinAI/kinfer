package engine

import "fmt"

// sizeState is what Open's sizing loop may make smaller after a measured
// context leaves too little memory: the conversations, the prefix pool, the
// context each conversation gets, the micro-batch, and the expert cache.
type sizeState struct {
	slots, prefix, perSeq, ub int
	shared                    bool
	moeCache                  int // MiB
}

// sizeRules is what that loop is allowed to change, and the floors it keeps.
type sizeRules struct {
	// What the operator left to choose: an explicit flag is never overridden.
	ubAuto, slotsAuto, prefixAuto, ctxAuto bool
	// cacheAuto is an expert cache placement chose, which may give up as
	// much as half of itself (to cacheFloor) to keep the prefix pool.
	cacheAuto   bool
	fixedLayout bool // -flex=false
	hostExperts bool
	floor       int // the context below which slots go first (ctxFloor)
	cacheFloor  int
	cut         int // MiB the expert cache would give up this time (cacheCut)
}

// shrink is one step smaller, and what it gave up, or false when nothing is
// left to give.
//
// Smaller the way AutoSize would have chosen had its estimate been right: a
// conversation goes before context does. This is where that matters most,
// because a hybrid's per-sequence state is what the estimate undercounts — and
// dropping a sequence frees a whole one of them. Halving alone took the box to
// 4 slots of 4096.
func shrink(s sizeState, r sizeRules) (sizeState, string, bool) {
	switch {
	// A bigger micro-batch is speed, not capacity: it goes before
	// anything a conversation would notice — unless experts are in host
	// memory, where every pass reads them again and it goes last.
	case r.ubAuto && s.ub > ubatchDefault && !r.hostExperts:
		s.ub = lowerUbatch(s.ub)
		return s, fmt.Sprintf("a micro-batch of %d instead", s.ub), true
	case s.perSeq/2 < r.floor && r.slotsAuto && s.slots > 1:
		s.slots /= 2
		s.prefix = min(s.prefix, max(s.slots/2, 1))
		return s, fmt.Sprintf("%d slots + %d prefix instead, keeping %d tokens each", s.slots, s.prefix, s.perSeq), true
	// When the context is the operator's, halving it is not ours to do, so
	// the pool goes first whatever the floor.
	case (s.perSeq/2 < r.floor || !r.ctxAuto) && r.prefixAuto && s.prefix > 0 && !s.shared:
		// The pool's own cells go before the pool does: sharing the slots'
		// cells, it still carries an agent's turns forward.
		s.shared = true
		return s, fmt.Sprintf("the prefix pool shares the slots' cells instead, keeping %d tokens each", s.perSeq), true
	// Placement sized the expert cache before this measurement saw what a
	// pool costs: a pool needs the KV cache unified, and a pass over all
	// of it — on the box, Flash-Next's 4 x 32768 took a pass of 5.8 GiB
	// where the conversations' own caches took 2.2. The cache gives that
	// up before the pool goes: a smaller cache costs some decode speed,
	// no pool reads every turn whole again (11.8 s for 7,800 tokens there).
	case (s.perSeq/2 < r.floor || !r.ctxAuto) && r.prefixAuto && s.prefix > 0 && r.cacheAuto && s.moeCache-r.cut >= r.cacheFloor:
		s.moeCache -= r.cut
		return s, fmt.Sprintf("the expert cache gives up %d MiB instead, keeping %d prefix with %d MiB of cache", r.cut, s.prefix, s.moeCache), true
	// Experts in host memory get passes of 4096 (ubatchHost), but a pool
	// needs the KV cache unified, and a pass reads all of its cells: on the
	// box, Flash-Next with -slots 4 lost its pool to that larger pass and
	// read every turn whole again, 19–41 s to a first token. 2048 is what
	// those models read prompts in before v0.2.5, and it goes before the
	// pool's last entry does — only then: taken sooner, on the MacBook
	// Pro it read Flash-Next's 11,000-token prompt at 660 tok/s where
	// v0.2.5, giving up a pool entry instead, read it at 820.
	case (s.perSeq/2 < r.floor || !r.ctxAuto) && r.prefixAuto && s.prefix == 1 && r.ubAuto && r.hostExperts && s.ub > ubatchLarge:
		s.ub = ubatchLarge
		return s, fmt.Sprintf("a micro-batch of %d instead, which keeps the pool's last entry", s.ub), true
	// The pool's last entry is worth more than the far half of every
	// slot's context: without it nothing is reused, and every turn is read
	// whole again. On the box, qwen3.8-27b loaded beside ornith with -slots
	// 4 gave its last entry up to keep 4 slots of 32768, and kinhome's
	// 13,000-token turns waited 43–50 s each for a first token where
	// ornith's, pooled, waited 0.4. A halved context is still there for a
	// prompt that needs it: flex gives that prompt the longer slot while it
	// runs. Not below poolCtxMin, and not when the layout is fixed.
	case r.prefixAuto && s.prefix == 1 && r.ctxAuto && !r.fixedLayout && s.perSeq/2 < r.floor && s.perSeq/2 >= poolCtxMin:
		s.perSeq /= 2
		return s, "halving the context instead, which keeps the pool's last entry", true
	case (s.perSeq/2 < r.floor || !r.ctxAuto) && r.prefixAuto && s.prefix > 0:
		s.prefix--
		return s, fmt.Sprintf("%d prefix instead, keeping %d tokens each", s.prefix, s.perSeq), true
	case r.ctxAuto && s.perSeq/2 >= 2048:
		s.perSeq /= 2
		return s, "halving the context", true
	case r.ubAuto && s.ub > ubatchDefault:
		s.ub = lowerUbatch(s.ub)
		return s, fmt.Sprintf("a micro-batch of %d instead", s.ub), true
	}
	return s, "", false
}
