//go:build !windows

package hardware

func windowsRAM() int64 { return 0 }
