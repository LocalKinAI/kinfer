package main

import (
	"fmt"
	"runtime"

	"github.com/LocalKinAI/kinfer/internal/nativelib"
)

// version is the release this binary was built as, set by scripts/release.sh
// with -ldflags "-X main.version=v0.1.0"; a build from a checkout says dev.
var version = "dev"

// cmdVersion prints the release and the llama.cpp libraries inside the
// binary: the release set, or the patched one that reads experts from the SSD
// at speed (scripts/build-libs.sh), which is what a bug report needs first.
func cmdVersion() error {
	set, err := nativelib.LibrarySet()
	if err != nil {
		set = err.Error()
	}
	fmt.Printf("kinfer %s, llama.cpp %s, %s/%s\n", version, set, runtime.GOOS, runtime.GOARCH)
	return nil
}
