package engine

import "testing"

// The box's layout: -ctx 16384 -slots 4 on a model trained for 262144.
var boxHome = layout{slots: 4, perSeq: 16384}

func TestFlexWantIsTheMostSlotsThatHoldThePromptWhole(t *testing.T) {
	f := newFlex(boxHome, 262144, nil)
	if f == nil {
		t.Fatal("no flex for 4 x 16384 on a model trained for 262144")
	}
	// Each slot keeps replyRoom for the answer: an eighth of it, or the
	// caller's num_predict when that is smaller. The boundaries are where a
	// prompt stops running whole — the same line fitMessages draws.
	cases := []struct {
		prompt, maxTokens, want int
	}{
		{100, 0, 4},
		{14336, 0, 4}, // 16384 - 2048
		{14337, 0, 3},
		{19040, 0, 3}, // 21760 - 2720
		{19041, 0, 2},
		{28672, 0, 2}, // 32768 - 4096
		{28673, 0, 1},
		{57344, 0, 1}, // 65536 - 8192
		{90000, 0, 1}, // held whole by nothing: the longest slot, fewest turns lost
		{15884, 500, 4},
		{15885, 500, 3},
		{14337, 32000, 3}, // Claude Code's max_tokens is capped like any other
	}
	for _, c := range cases {
		if got := f.want(c.prompt, c.maxTokens); got != c.want {
			t.Errorf("want(%d tokens, num_predict %d) = %d slots, want %d", c.prompt, c.maxTokens, got, c.want)
		}
	}
}

func TestFlexLayoutsHoldTheSameTokens(t *testing.T) {
	f := newFlex(boxHome, 262144, nil)
	for n, per := range map[int]int{4: 16384, 3: 21760, 2: 32768, 1: 65536} {
		l := f.layout(n)
		if l.slots != n || l.perSeq != per {
			t.Errorf("layout(%d) = %s, want %d x %d", n, l, n, per)
		}
		if l.slots*l.perSeq > f.total {
			t.Errorf("layout(%d) holds %d tokens, more than home's %d", n, l.slots*l.perSeq, f.total)
		}
	}
	if got, want := f.offers(), "3 x 21760, 2 x 32768, 1 x 65536"; got != want {
		t.Errorf("offers = %q, want %q", got, want)
	}
}

func TestFlexHomeKeepsItsPoolAndItsExactContext(t *testing.T) {
	// A -ctx that is not a multiple of 256 is still what home gets: it is the
	// context the model loaded with, and rounding it would rebuild nothing.
	home := layout{slots: 3, prefix: 1, perSeq: 10000, shared: true}
	f := newFlex(home, 0, nil)
	if got := f.layout(3); got != home {
		t.Errorf("layout(home) = %+v, want %+v", got, home)
	}
	if got := f.layout(1); got.prefix != 0 || got.shared || got.perSeq != 29952 {
		t.Errorf("layout(1) = %+v, want 1 x 29952 with no pool", got)
	}
}

func TestFlexIsOffWhenNoLayoutWouldBeLonger(t *testing.T) {
	if f := newFlex(layout{slots: 1, perSeq: 65536}, 262144, nil); f != nil {
		t.Error("one slot has nothing to give up, but got a flex")
	}
	if f := newFlex(boxHome, 16384, nil); f != nil {
		t.Error("a model trained for 16384 gains nothing from fewer slots, but got a flex")
	}
	if f := newFlex(boxHome, 0, nil); f == nil {
		t.Error("an unknown trained context should not switch flex off")
	}
}

func TestFlexStopsAtTheTrainedContext(t *testing.T) {
	// Trained for 32768: two slots already reach it, so one slot is not a
	// longer conversation, only a lonelier one.
	f := newFlex(boxHome, 32768, nil)
	if got, want := f.offers(), "3 x 21760, 2 x 32768"; got != want {
		t.Errorf("offers = %q, want %q", got, want)
	}
	if got := f.want(28673, 0); got != 2 {
		t.Errorf("a prompt no layout holds whole got %d slots, want 2: the longest slot, with the most company", got)
	}
}

func TestFlexSkipsRefusedLayouts(t *testing.T) {
	f := newFlex(boxHome, 262144, nil)
	f.refused[1] = true
	if got := f.want(40000, 0); got != 2 {
		t.Errorf("with one slot refused, a 40000-token prompt got %d slots, want 2", got)
	}
	f.refused[2], f.refused[3] = true, true
	if got := f.want(40000, 0); got != 4 {
		t.Errorf("with every other layout refused, got %d slots, want home's 4", got)
	}
}

// sched is a scheduler with n slots, the first len(running) of them busy with
// jobs that want those slot counts. Enough for place, which reads nothing else.
func sched(n int, fx *flex, running ...int) *scheduler {
	s := &scheduler{flex: fx, slots: make([]*slot, n)}
	for i := range s.slots {
		s.slots[i] = &slot{seq: int32(i), logitIdx: -1}
	}
	for i, w := range running {
		s.slots[i].job = &job{want: w}
	}
	return s
}

func TestPlaceFollowsTheHeadOfTheQueue(t *testing.T) {
	fx := newFlex(boxHome, 262144, nil)
	cases := []struct {
		name    string
		s       *scheduler
		want    int
		outcome placement
	}{
		{"fits home, a slot is free", sched(4, fx), 4, placeNow},
		{"fits home, every slot busy", sched(4, fx, 4, 4, 4, 4), 4, placeWait},
		{"too long for home: the slots drain first", sched(4, fx, 4, 4), 1, placeWait},
		{"too long for home, and the slots are idle", sched(4, fx), 1, placeReshape},
		{"short, behind the one long conversation", sched(1, fx, 1), 4, placeWait},
		{"short, a free longer slot while the long one runs", sched(2, fx, 2), 4, placeNow},
		{"short, and nothing running needs the longer slots", sched(2, fx, 4), 4, placeWait},
		{"short, the longer slots idle", sched(2, fx), 4, placeReshape},
		{"longer still than the longer slots", sched(2, fx, 2), 1, placeWait},
		{"a fixed layout never reshapes", sched(4, nil, 4, 4, 4), 1, placeNow},
		{"a fixed layout waits for a slot", sched(4, nil, 4, 4, 4, 4), 4, placeWait},
	}
	for _, c := range cases {
		if got := c.s.place(&job{want: c.want}); got != c.outcome {
			t.Errorf("%s: place = %d, want %d", c.name, got, c.outcome)
		}
	}
}

func TestWaitingCountsTheHeldRequest(t *testing.T) {
	s := &scheduler{incoming: make(chan *job, 4)}
	s.incoming <- &job{}
	if got := s.Waiting(); got != 1 {
		t.Fatalf("Waiting = %d with one queued, want 1", got)
	}
	s.hold(&job{})
	if got := s.Waiting(); got != 2 {
		t.Errorf("Waiting = %d with one queued and one held for its layout, want 2", got)
	}
	s.hold(nil)
	if got := s.Waiting(); got != 1 {
		t.Errorf("Waiting = %d after the held one took a slot, want 1", got)
	}
}
