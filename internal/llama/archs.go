package llama

import (
	"bytes"
	"os"
	"path/filepath"
	"sync"
)

// Supports reports whether the embedded llama.cpp can run a model architecture,
// and whether that answer is known at all.
//
// llama.cpp exposes no API for this, and the answer belongs to the exact build
// shipped inside the binary rather than to llama.cpp in general. So the library
// itself is searched for the architecture name as a standalone string, which is
// how llama-arch.cpp stores it.
//
// Searching the whole file rather than walking the table matters. The names are
// mostly laid out contiguously, but not all of them — following the run finds
// about half, and reporting the rest as unsupported would tell someone a model
// cannot run when it can. Searching cannot make that mistake. It can err the
// other way, claiming support for a name that appears in the binary for some
// other reason, and that is the better error: the model then fails to load with
// llama.cpp's own message rather than never being tried.
func Supports(dir, arch string) (supported, known bool) {
	data := libraryBytes(dir)
	if len(data) == 0 {
		return false, false
	}
	// If a name every build has cannot be found, the search itself is broken
	// and no answer should be given.
	if !hasCString(data, "llama") {
		return false, false
	}
	return hasCString(data, arch), true
}

func hasCString(data []byte, s string) bool {
	if s == "" {
		return false
	}
	needle := make([]byte, 0, len(s)+2)
	needle = append(needle, 0)
	needle = append(needle, s...)
	needle = append(needle, 0)
	return bytes.Contains(data, needle)
}

var (
	libOnce  sync.Once
	libBytes []byte
)

// libraryBytes reads the unpacked library once. It is tens of megabytes, held
// for the life of the process only when something asks about architectures.
func libraryBytes(dir string) []byte {
	libOnce.Do(func() {
		libBytes, _ = os.ReadFile(filepath.Join(dir, libName()))
	})
	return libBytes
}
