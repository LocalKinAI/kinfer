//go:build !windows

package dl

import "github.com/ebitengine/purego"

// Open loads the library at path and returns its handle.
//
// RTLD_GLOBAL so symbols resolve against the copy the backend already uses: two
// independently loaded copies of a library would each carry their own state and
// hand out handles the other cannot interpret.
func Open(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}

// Sym returns the address of name in the library, or an error when it has none.
func Sym(lib uintptr, name string) (uintptr, error) {
	return purego.Dlsym(lib, name)
}
