package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"runtime"

	"github.com/LocalKinAI/kinfer/internal/link"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
	"github.com/LocalKinAI/kinfer/internal/store"
)

// cmdWorker lends this machine's GPU to a kinfer serve on another, which puts
// some of a model's layers on it (serve -peers). See internal/llama/rpc.go.
//
// The protocol has no authentication: anyone who reaches the port can run
// computations and read what is loaded. So a worker listens on the cable
// between two machines and nowhere else — by default this Mac's end of a
// Thunderbolt cable (internal/link), where it also answers the machine at the
// other end when it asks for workers; otherwise a cable's address given by
// hand, or loopback. Never a LAN or Wi-Fi address.
func cmdWorker(args []string) error {
	fs := flag.NewFlagSet("worker", flag.ExitOnError)
	addr := fs.String("addr", "auto", "address to serve on: this machine's end of the cable to the other one (auto = the Thunderbolt Bridge's)")
	home, _ := os.UserHomeDir()
	cache := fs.String("cache", filepath.Join(home, ".kinfer", "rpc-cache"),
		"where to keep the weights a serve sends, so its next load sends hashes instead (\"\" to keep none)")
	threads := fs.Int("threads", runtime.NumCPU(), "CPU threads for work the GPU does not take")
	on := fs.String("on", "", "start the worker on another Mac instead, over ssh, in the background there: user@host, or a user alone for the Mac at the other end of the Thunderbolt cable")
	stop := fs.Bool("stop", false, "with -on, stop the worker running there")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("worker takes no arguments; %q was one", fs.Arg(0))
	}
	if *on != "" {
		return remoteWorker(*on, *stop)
	}
	if *stop {
		return fmt.Errorf("-stop stops a worker started with -on; a worker here stops with Ctrl-C")
	}
	bridge, berr := link.FindBridge()
	if *addr == "auto" {
		if berr != nil {
			return fmt.Errorf("%w\n       connect one, or give -addr: another cable's address, or 127.0.0.1:%d for this machine alone",
				berr, link.Port)
		}
		*addr = bridge.Addr()
	}
	// Found by the other machine only on the bridge: a worker given another
	// address is named by hand there too.
	onBridge := false
	if host, _, err := net.SplitHostPort(*addr); err == nil && berr == nil {
		onBridge = net.ParseIP(host).Equal(bridge.IP)
	}
	if *cache != "" {
		if err := os.MkdirAll(*cache, 0o755); err != nil {
			return err
		}
	}

	// The other machine places layers here by the free memory this worker
	// reports, and Metal counts only this process's allocations: the GPU
	// memory this machine's own apps hold would be offered too, and the
	// layers paged in and out of the GPU every token (V4.1 from the box,
	// 12 layers on the MacBook Pro: 2.3 tokens a second). The patched
	// llama.cpp reports what the whole GPU has left instead.
	os.Setenv("GGML_METAL_FREE_DEVICE", "1")
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
	// A worker waits between graphs on every token, while the other machine
	// computes its layers; see internal/llama/warm.go.
	if llama.KeepGPUWarm(llama.WarmWindowMs) {
		fmt.Printf("  the GPU is kept at speed between tokens\n")
	}
	if onBridge {
		go func() {
			if err := link.Answer(context.Background(), bridge, *addr); err != nil && !errors.Is(err, context.Canceled) {
				log.Printf("worker: the other machine cannot find this worker by itself (%v); name it there with -peers %s", err, *addr)
			}
		}()
		fmt.Printf("  on the other machine, over the Thunderbolt cable: kinfer run <model> finds this worker by itself "+
			"when the model does not fit there; kinfer serve -peers auto does the same, -peers %s names it\n", *addr)
	} else {
		fmt.Printf("  on the other machine: kinfer run -peers %s <model>, or kinfer serve -peers %s\n", *addr, *addr)
	}
	fmt.Printf("  no authentication — keep this address on a cable between two machines\n\n")
	return llama.ServeRPC(*addr, *cache, *threads)
}
