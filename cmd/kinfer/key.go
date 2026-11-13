package main

import (
	"net/http"
	"os"
	"time"
)

// keyed carries kinfer's own requests to a kinfer serve (ps, service, launch,
// run): it sends the server's key, KINFER_API_KEY, when that is set, as the
// server asks of a request from another machine. Downloads never go through
// it, so the key goes to kinfer and nowhere else.
type keyed struct{}

func (keyed) RoundTrip(r *http.Request) (*http.Response, error) {
	if k := os.Getenv("KINFER_API_KEY"); k != "" && r.Header.Get("Authorization") == "" {
		r = r.Clone(r.Context())
		r.Header.Set("Authorization", "Bearer "+k)
	}
	return http.DefaultTransport.RoundTrip(r)
}

// kinferClient is a client for a kinfer serve; 0 is no time limit.
func kinferClient(timeout time.Duration) *http.Client {
	return &http.Client{Timeout: timeout, Transport: keyed{}}
}
