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
	go func() {
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
	}()
	return ln.Addr().String()
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
