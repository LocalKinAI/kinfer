//go:build darwin

package main

import (
	"fmt"
	"github.com/ebitengine/purego"
)

// macOS QoS classes (sys/qos.h). Higher means the scheduler prefers a
// performance core; the lower ones are steered onto efficiency cores.
const (
	qosUserInteractive = 0x21
	qosUserInitiated   = 0x19
	qosDefault         = 0x15
	qosUtility         = 0x11
	qosBackground      = 0x09
)

var qosNames = map[uint32]string{
	qosUserInteractive: "user-interactive", qosUserInitiated: "user-initiated",
	qosDefault: "default", qosUtility: "utility", qosBackground: "background", 0: "unspecified",
}

var qosSelf func() uint32

func initQoS() {
	lib, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return
	}
	purego.RegisterLibFunc(&qosSelf, lib, "qos_class_self")
}

// qosReport says what QoS the calling thread runs at.
func qosReport() string {
	if qosSelf == nil {
		return "unknown"
	}
	q := qosSelf()
	if n, ok := qosNames[q]; ok {
		return n
	}
	return fmt.Sprintf("0x%x", q)
}
