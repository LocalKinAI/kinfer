package llama

import (
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"path/filepath"
	"time"

	"github.com/ebitengine/purego"
)

// A model across machines, through ggml's RPC backend.
//
// One machine runs a worker: its GPU, offered over TCP to whoever connects.
// Another loads a model with that worker's device in its device list, and
// llama.cpp puts some of the layers there. Each token's activations cross the
// wire between the two halves; the weights cross once, at load, unless the
// worker kept them in its cache from an earlier load.
//
// Both ends must be the same llama.cpp build — the protocol version is checked
// at connect, and moved from 7 to 8 between b11175 and b11998 — which is why the
// worker is kinfer itself rather than llama.cpp's rpc-server: one binary on
// every machine cannot disagree with itself.
//
// The protocol has no authentication and no encryption, and llama.cpp says so.
// A worker belongs on a cable between two machines, not on a network.

var (
	rpcStartServer func(endpoint, cacheDir *byte, nThreads, nDevices uint64, devices *uintptr)
	rpcAddServer   func(endpoint *byte) uintptr
	regDevCount    func(reg uintptr) uint64
	regDevGet      func(reg uintptr, i uint64) uintptr
)

// bindRPC resolves the RPC backend's entry points, which live in libggml-rpc
// (and the registry's in libggml-base). Like bindGGML it is best-effort: a
// build without them serves one machine and says so when asked for more.
func bindRPC(dir string) {
	rpc, err := purego.Dlopen(siblingLib(dir, "ggml-rpc"), purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return
	}
	base, err := purego.Dlopen(siblingLib(dir, "ggml-base"), purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return
	}
	for _, b := range []struct {
		fn   any
		lib  uintptr
		name string
	}{
		{&rpcStartServer, rpc, "ggml_backend_rpc_start_server"},
		{&rpcAddServer, rpc, "ggml_backend_rpc_add_server"},
		{&regDevCount, base, "ggml_backend_reg_dev_count"},
		{&regDevGet, base, "ggml_backend_reg_dev_get"},
	} {
		if err := register(b.fn, b.lib, b.name); err != nil {
			rpcStartServer = nil // all or nothing
			return
		}
	}
}

// siblingLib names one of libllama's sibling libraries the way this platform
// spells it, as ggmlName does for libggml.
func siblingLib(dir, stem string) string {
	switch libName() {
	case "llama.dll":
		return filepath.Join(dir, stem+".dll")
	case "libllama.so.0":
		return filepath.Join(dir, "lib"+stem+".so.0")
	default:
		return filepath.Join(dir, "lib"+stem+".0.dylib")
	}
}

// LocalGPUs returns this machine's GPU devices, in registry order.
func LocalGPUs() []uintptr {
	if devCount == nil {
		return nil
	}
	var out []uintptr
	for i := uint64(0); i < devCount(); i++ {
		d := devGet(i)
		if t := DeviceType(devType(d)); d != 0 && (t == DeviceGPU || t == DeviceIGPU) {
			out = append(out, d)
		}
	}
	return out
}

// ServeRPC offers this machine's GPUs to RPC clients at endpoint ("host:port")
// and does not return while it serves. cacheDir, when set, keeps every tensor a
// client sends, by hash, so the next load of the same model sends hashes
// instead of weights.
func ServeRPC(endpoint, cacheDir string, threads int) error {
	if rpcStartServer == nil {
		return fmt.Errorf("this build has no RPC backend")
	}
	devs := LocalGPUs()
	if len(devs) == 0 {
		return fmt.Errorf("no GPU to offer")
	}
	ep := append([]byte(endpoint), 0)
	var cache *byte
	if cacheDir != "" {
		c := append([]byte(cacheDir), 0)
		cache = &c[0]
	}
	rpcStartServer(&ep[0], cache, uint64(threads), uint64(len(devs)), &devs[0])
	// It returns only when it could not listen.
	return fmt.Errorf("the RPC server at %s stopped (could it bind that address?)", endpoint)
}

// The protocol of the llama.cpp kinfer embeds (ggml-rpc.h, b11998), as much of
// it as a hello takes: a command byte and a length-prefixed request of the
// connection capabilities, all zero for plain TCP; a length-prefixed response
// of the server's version and its capabilities.
const (
	rpcCmdHello     = 14
	rpcCapsSize     = 24
	RPCProtoVersion = 8
)

// RPCHello says whether a worker at endpoint answers a hello within wait, in
// this build's protocol, and hangs up. llama.cpp's own connection does not ask
// first: a worker serving another machine takes one connection at a time, and
// llama.cpp waits on it forever; one of another version, and it aborts the
// whole process.
func RPCHello(endpoint string, wait time.Duration) error {
	c, err := net.DialTimeout("tcp", endpoint, min(wait, 3*time.Second))
	if err != nil {
		return fmt.Errorf("worker %s is not reachable: %w", endpoint, err)
	}
	defer c.Close()
	_ = c.SetDeadline(time.Now().Add(wait))
	req := make([]byte, 1+8+rpcCapsSize)
	req[0] = rpcCmdHello
	binary.LittleEndian.PutUint64(req[1:9], rpcCapsSize)
	if _, err := c.Write(req); err != nil {
		return fmt.Errorf("worker %s is not reachable: %w", endpoint, err)
	}
	rsp := make([]byte, 8+4+rpcCapsSize)
	if _, err := io.ReadFull(c, rsp); err != nil {
		return fmt.Errorf("worker %s did not answer within %s — is it serving another machine?", endpoint, wait)
	}
	if n := binary.LittleEndian.Uint64(rsp[:8]); n != 4+rpcCapsSize || rsp[8] != RPCProtoVersion {
		return fmt.Errorf("worker %s speaks RPC protocol %d.%d, this kinfer %d: run the same kinfer on both machines",
			endpoint, rsp[8], rsp[9], RPCProtoVersion)
	}
	return nil
}

// RPCDevices connects to a worker and returns its devices, for a model's
// device list. A worker that is not there, or speaks another protocol
// version, offers none.
func RPCDevices(endpoint string) ([]uintptr, error) {
	if rpcAddServer == nil {
		return nil, fmt.Errorf("this build has no RPC backend")
	}
	ep := append([]byte(endpoint), 0)
	reg := rpcAddServer(&ep[0])
	if reg == 0 {
		return nil, fmt.Errorf("no devices at %s: is a kinfer worker of this same build running there, and reachable?", endpoint)
	}
	n := regDevCount(reg)
	out := make([]uintptr, 0, n)
	for i := uint64(0); i < n; i++ {
		if d := regDevGet(reg, i); d != 0 {
			out = append(out, d)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("the worker at %s offers no devices", endpoint)
	}
	return out, nil
}

// DeviceName is a device handle's name, such as "MTL0" or "RPC0".
func DeviceName(d uintptr) string {
	if devName == nil || d == 0 {
		return ""
	}
	return goString(devName(d))
}

// DeviceMemory is what a device handle reports free and in all — for a remote
// device, what its worker reports.
func DeviceMemory(d uintptr) (free, total uint64) {
	if devMemory == nil || d == 0 {
		return 0, 0
	}
	devMemory(d, &free, &total)
	return free, total
}
