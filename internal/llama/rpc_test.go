package llama

import (
	"encoding/binary"
	"io"
	"net"
	"strings"
	"testing"
	"time"
)

// fakeWorker answers one hello per connection as a worker of protocol major
// would; with major 0 it reads the hello and never answers, as a worker busy
// with another machine does.
func fakeWorker(t *testing.T, major byte) string {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go answerHellos(ln, major)
	return ln.Addr().String()
}

// answerHellos is fakeWorker's accept loop, until ln is closed.
func answerHellos(ln net.Listener, major byte) {
	for {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		go func() {
			defer c.Close()
			req := make([]byte, 1+8+rpcCapsSize)
			if _, err := io.ReadFull(c, req); err != nil || req[0] != rpcCmdHello {
				return
			}
			if major == 0 {
				time.Sleep(time.Second)
				return
			}
			rsp := make([]byte, 8+4+rpcCapsSize)
			binary.LittleEndian.PutUint64(rsp, 4+rpcCapsSize)
			rsp[8] = major
			c.Write(rsp)
		}()
	}
}

func TestRPCHello(t *testing.T) {
	if err := RPCHello(fakeWorker(t, RPCProtoVersion), time.Second); err != nil {
		t.Errorf("a worker of this protocol: %v", err)
	}
	if err := RPCHello(fakeWorker(t, 7), time.Second); err == nil || !strings.Contains(err.Error(), "protocol 7") {
		t.Errorf("a worker of protocol 7: %v", err)
	}
	if err := RPCHello(fakeWorker(t, 0), 200*time.Millisecond); err == nil || !strings.Contains(err.Error(), "did not answer") {
		t.Errorf("a busy worker: %v", err)
	}
}

// A worker turning from its last connection refuses the next for a moment;
// one that is not there at all is refused for good.
func TestRPCHelloTriesAgain(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close()
	late := make(chan net.Listener, 1)
	time.AfterFunc(300*time.Millisecond, func() {
		ln, err := net.Listen("tcp", addr)
		if err != nil {
			late <- nil
			return
		}
		late <- ln
		answerHellos(ln, RPCProtoVersion)
	})
	t.Cleanup(func() {
		if ln := <-late; ln != nil {
			ln.Close()
		}
	})
	if err := RPCHello(addr, time.Second); err != nil {
		t.Errorf("a worker listening 300 ms late: %v", err)
	}

	gone, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr = gone.Addr().String()
	gone.Close()
	start := time.Now()
	if err := RPCHello(addr, time.Second); err == nil || !strings.Contains(err.Error(), "not reachable") {
		t.Errorf("no worker: %v", err)
	}
	if d := time.Since(start); d < 1500*time.Millisecond || d > 4*time.Second {
		t.Errorf("no worker failed after %v, want about 2 s", d)
	}
}
