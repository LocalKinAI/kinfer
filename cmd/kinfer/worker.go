package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
	"github.com/LocalKinAI/kinfer/internal/store"
)

// cmdWorker lends this machine's GPU to a kinfer serve on another, which puts
// some of a model's layers on it (serve -peers). See internal/llama/rpc.go.
//
// The default address is loopback because the protocol has no authentication:
// anyone who reaches the port can run computations and read what is loaded. A
// worker meant for another machine is given the address of the cable between
// them — a Thunderbolt bridge, a USB link — never a LAN or Wi-Fi address.
func cmdWorker(args []string) error {
	fs := flag.NewFlagSet("worker", flag.ExitOnError)
	addr := fs.String("addr", "127.0.0.1:50052", "address to serve on: this machine's end of the cable to the other one")
	home, _ := os.UserHomeDir()
	cache := fs.String("cache", filepath.Join(home, ".kinfer", "rpc-cache"),
		"where to keep the weights a serve sends, so its next load sends hashes instead (\"\" to keep none)")
	threads := fs.Int("threads", runtime.NumCPU(), "CPU threads for work the GPU does not take")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("worker takes no arguments; %q was one", fs.Arg(0))
	}
	if *cache != "" {
		if err := os.MkdirAll(*cache, 0o755); err != nil {
			return err
		}
	}

	dir, err := nativelib.Prepare()
	if err != nil {
		return err
	}
	if err := llama.Bind(dir); err != nil {
		return err
	}
	gpus := llama.LocalGPUs()
	if len(gpus) == 0 {
		return fmt.Errorf("no GPU here to lend")
	}
	for _, g := range gpus {
		free, total := llama.DeviceMemory(g)
		fmt.Printf("kinfer worker on %s — lending %s, %s free of %s\n", *addr, llama.DeviceName(g),
			store.HumanSize(int64(free)), store.HumanSize(int64(total)))
	}
	if *cache != "" {
		fmt.Printf("  weights kept in %s, so a model's second load sends hashes, not tensors\n", *cache)
	}
	fmt.Printf("  on the other machine: kinfer serve -peers %s\n", *addr)
	fmt.Printf("  no authentication — keep this address on a cable between two machines\n\n")
	return llama.ServeRPC(*addr, *cache, *threads)
}
