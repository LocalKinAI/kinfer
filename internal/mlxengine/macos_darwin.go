package mlxengine

import (
	"strconv"
	"strings"
	"syscall"
)

// macOS is this Mac's version, from kern.osproductversion.
func macOS() (major, minor int, ok bool) {
	v, err := syscall.Sysctl("kern.osproductversion")
	if err != nil {
		return 0, 0, false
	}
	parts := strings.Split(v, ".")
	major, err = strconv.Atoi(parts[0])
	if err != nil {
		return 0, 0, false
	}
	if len(parts) > 1 {
		minor, _ = strconv.Atoi(parts[1])
	}
	return major, minor, true
}
