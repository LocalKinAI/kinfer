//go:build !darwin

package main

// Thread QoS is Apple's; elsewhere there is nothing to report.

func initQoS() {}

func qosReport() string { return "not applicable" }
