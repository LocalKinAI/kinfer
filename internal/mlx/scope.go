package mlx

// Scope collects the arrays made while it is open so one Close frees them all.
// A forward pass makes hundreds of intermediates per step; tracking them here
// is what keeps the model code free of a Free after every line.
//
// Scopes nest: the innermost open one collects. Arrays made with no scope open
// are the caller's to free — weights are loaded that way.
type Scope struct {
	arrays []Array
	kept   map[Array]bool
	parent *Scope
}

var current *Scope

// NewScope opens a scope; every array made until it closes belongs to it.
func NewScope() *Scope {
	s := &Scope{parent: current}
	current = s
	return s
}

// Keep exempts a from this scope's Close, handing it to the enclosing scope
// (or to the caller, with none open). The result of a step is kept this way.
func (s *Scope) Keep(arrays ...Array) {
	if s.kept == nil {
		s.kept = make(map[Array]bool)
	}
	for _, a := range arrays {
		s.kept[a] = true
		if s.parent != nil {
			s.parent.arrays = append(s.parent.arrays, a)
		}
	}
}

// Escape exempts a from this scope's Close without handing it to any other
// scope: the caller owns it and frees it. State that outlives every step — a
// model's cache — leaves its scope this way.
func (s *Scope) Escape(arrays ...Array) {
	if s.kept == nil {
		s.kept = make(map[Array]bool)
	}
	for _, a := range arrays {
		s.kept[a] = true
	}
}

// Close frees every array the scope collected and did not keep.
func (s *Scope) Close() {
	for _, a := range s.arrays {
		if !s.kept[a] {
			arrayFree(a)
		}
	}
	s.arrays = nil
	if current == s {
		current = s.parent
	}
}

func track(a Array) {
	if current != nil {
		current.arrays = append(current.arrays, a)
	}
}
