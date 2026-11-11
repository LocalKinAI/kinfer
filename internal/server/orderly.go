package server

import (
	"io"
	"net"
	"net/http"
	"sync"
	"time"
)

// A connection closed the moment its reply is written lost the reply on the
// box: a client that asks for Connection: close — Python's urllib always does,
// and curl when told to — got nothing back from any server there, Ollama
// included, and the request was gone without a trace (http.client's
// RemoteDisconnected). From the mini, 8 of 10 replies to such requests were
// lost; the same server sleeping 0.2 s before the close, or shutting its
// writing half and waiting for the client to close first, lost none. Nothing
// was lost from the box's own processes or on the other Macs: the box runs
// network filters of its own (a VPN client's network extension, an endpoint
// sensor), which is where the reply is dropped.
//
// So a connection to these servers is closed the orderly way, as net/http
// itself does when it has to (closeWriteAndWait): the writing half first,
// which sends what is written and then the end of it; then whatever the client
// still sends is read, for up to lingerClose, until it closes its own half;
// then the socket. The client waits for none of this: its reply and the end of
// it reach it at once.
const lingerClose = 2 * time.Second

// OrderlyListener closes the connections it accepts the orderly way.
func OrderlyListener(l net.Listener) net.Listener { return orderlyListener{l} }

type orderlyListener struct{ net.Listener }

func (l orderlyListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	if tc, ok := c.(*net.TCPConn); ok {
		return &orderlyConn{TCPConn: tc}, nil
	}
	return c, nil
}

type orderlyConn struct {
	*net.TCPConn
	once sync.Once
	err  error
}

// Close shuts the writing half, and closes the socket once the client has
// closed its own or lingerClose has passed, in the background. net/http may
// close a connection from more than one goroutine; only the first counts.
func (c *orderlyConn) Close() error {
	c.once.Do(func() {
		if err := c.TCPConn.CloseWrite(); err != nil {
			c.err = c.TCPConn.Close()
			return
		}
		go func() {
			c.TCPConn.SetReadDeadline(time.Now().Add(lingerClose))
			io.Copy(io.Discard, c.TCPConn)
			c.TCPConn.Close()
		}()
	})
	return c.err
}

// ListenAndServeOrderly is http.ListenAndServe on an OrderlyListener.
func ListenAndServeOrderly(addr string, h http.Handler) error {
	l, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	return (&http.Server{Handler: h, ReadHeaderTimeout: 10 * time.Second}).Serve(OrderlyListener(l))
}
