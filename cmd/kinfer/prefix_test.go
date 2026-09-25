package main

import (
	"flag"
	"testing"
)

// -prefix says what it means. It was an int where 0 meant "sized with the
// rest": on the box, `-prefix 0` typed to turn the pool off sized it at four
// entries instead, and the first request failed with no memory left.
func TestPrefixFlagSaysWhatItMeans(t *testing.T) {
	for in, want := range map[string]int{
		"auto": 0, "AUTO": 0, "off": -1, "none": -1, "0": -1, "-1": -1, "3": 3, " 2 ": 2,
	} {
		var p prefixFlag
		if err := p.Set(in); err != nil {
			t.Errorf("-prefix %q refused: %v", in, err)
			continue
		}
		if int(p) != want {
			t.Errorf("-prefix %q = %d, want %d", in, int(p), want)
		}
	}
	var p prefixFlag
	if err := p.Set("lots"); err == nil {
		t.Error(`-prefix "lots" was accepted`)
	}
	if got := p.String(); got != "auto" {
		t.Errorf("the default prints as %q in -h, want \"auto\"", got)
	}

	fs, o := serveFlags(flag.ContinueOnError)
	if err := fs.Parse(nil); err != nil || o.prefix != 0 {
		t.Errorf("unset -prefix = %v (%v), want auto", o.prefix, err)
	}
	fs, o = serveFlags(flag.ContinueOnError)
	if err := fs.Parse([]string{"-prefix", "off"}); err != nil || o.prefix != -1 {
		t.Errorf("-prefix off = %v (%v), want off", o.prefix, err)
	}
	// `kinfer install` records flags verbatim; nonsense must be refused at the
	// keyboard, not by a launchd job that fails at every login.
	if err := checkServeFlags([]string{"-prefix", "lots"}); err == nil {
		t.Error("install accepted -prefix lots")
	}
	if err := checkServeFlags([]string{"-addr", ":11590", "-ctx", "16384", "-slots", "4", "-prefix", "-1"}); err != nil {
		t.Errorf("the flags recorded on the box no longer parse: %v", err)
	}
}
