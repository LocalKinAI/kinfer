package main

import (
	"runtime"
	"strings"
	"testing"
)

func TestParseDriverCUDA(t *testing.T) {
	out := "| NVIDIA-SMI 570.86.15   Driver Version: 570.86.15   CUDA Version: 12.8     |"
	if v, ok := parseDriverCUDA(out); !ok || v != 12 {
		t.Errorf("got %d, %v", v, ok)
	}
	if _, ok := parseDriverCUDA("no such thing"); ok {
		t.Error("a version was read from nothing")
	}
}

func TestPickCUDATakesTheNewestTheDriverCanRun(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skip("the Linux x86-64 table")
	}
	for driver, want := range map[int]string{13: "13.4", 14: "13.4", 12: "12.8"} {
		b, err := pickCUDA(driver)
		if err != nil || b.Variant != want {
			t.Errorf("driver CUDA %d: got %s, %v; want %s", driver, b.Variant, err, want)
		}
	}
	if _, err := pickCUDA(11); err == nil || !strings.Contains(err.Error(), "older") {
		t.Errorf("driver CUDA 11: %v", err)
	}
}
