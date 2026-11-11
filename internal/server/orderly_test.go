package server

import (
	"bufio"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"
)

// A client that asks for Connection: close gets its reply and then the end of
// the connection; one that keeps the connection alive can send another.
func TestOrderlyListenerAnswersThenCloses(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		io.WriteString(w, "ok")
	})}
	go srv.Serve(OrderlyListener(l))
	defer srv.Close()

	c, err := net.Dial("tcp", l.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	c.SetDeadline(time.Now().Add(5 * time.Second))
	io.WriteString(c, "POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 4\r\nConnection: close\r\n\r\nbody")
	all, err := io.ReadAll(c) // to the server's end of the connection
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(all), "HTTP/1.1 200") || !strings.HasSuffix(string(all), "ok") {
		t.Fatalf("got %q", all)
	}

	k, err := net.Dial("tcp", l.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer k.Close()
	k.SetDeadline(time.Now().Add(5 * time.Second))
	br := bufio.NewReader(k)
	for i := 0; i < 2; i++ {
		io.WriteString(k, "GET / HTTP/1.1\r\nHost: x\r\n\r\n")
		resp, err := http.ReadResponse(br, nil)
		if err != nil {
			t.Fatalf("request %d on a kept connection: %v", i, err)
		}
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if string(b) != "ok" {
			t.Fatalf("request %d: %q", i, b)
		}
	}
}
