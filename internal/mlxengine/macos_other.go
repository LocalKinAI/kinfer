//go:build !darwin

package mlxengine

// macOS is unknown off a Mac, where MLX does not run anyway.
func macOS() (major, minor int, ok bool) { return 0, 0, false }
