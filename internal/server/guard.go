package server

import (
	"crypto/subtle"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"strings"
)

// ─── the door ────────────────────────────────────────────────────────────────
//
// kinfer serve answers whatever reaches its port. Two things reached it that
// should not have:
//
//   - A web page in a browser on the same network, or on the machine itself,
//     could have the browser send requests: load a model, or — from a browser
//     on the machine — stop or adopt a service. The browser could not read
//     the answer, but the request was made. A request from a web page carries
//     its page's Origin; one from a page that is not this machine's own, nor
//     an app's, is refused. KINFER_ORIGINS adds pages (comma-separated, *
//     matching any part: "http://192.168.0.*:7700").
//   - Anyone on the network could use it. KINFER_API_KEY, when set, is asked
//     of every request from another machine (Authorization: Bearer, or
//     x-api-key as Anthropic's clients send it); this machine's own, and
//     /api/version and /health, which say nothing, are not asked. Unset, as
//     before: the network is trusted.

// guard is the door every endpoint is behind.
func (s *Server) guard(next http.Handler) http.Handler {
	key := os.Getenv("KINFER_API_KEY")
	extra := splitList(os.Getenv("KINFER_ORIGINS"))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if o := r.Header.Get("Origin"); o != "" && !originAllowed(o, r.Host, extra) {
			writeJSON(w, http.StatusForbidden, map[string]string{"error": "this kinfer server refuses requests from the web page at " + o +
				"; KINFER_ORIGINS lists the pages it allows"})
			return
		}
		if key != "" && !fromThisMachine(r) && r.URL.Path != "/api/version" && r.URL.Path != "/health" && !keyGiven(r, key) {
			w.Header().Set("WWW-Authenticate", `Bearer realm="kinfer"`)
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "this kinfer server asks for its key (KINFER_API_KEY), " +
				"sent as Authorization: Bearer KEY or as x-api-key: KEY"})
			return
		}
		next.ServeHTTP(w, r)
	})
}

// appSchemes are the pages of apps, not of the web: an Electron or Tauri
// app's own, a file opened in a browser, an editor's webview — Ollama lets
// the same ones through.
var appSchemes = map[string]bool{"app": true, "file": true, "tauri": true, "vscode-webview": true, "vscode-file": true}

// originAllowed reports a page allowed to send requests: one of this machine
// (localhost, a loopback address, 0.0.0.0), one served from the address the
// request is for, an app's, or one KINFER_ORIGINS lists.
func originAllowed(origin, host string, extra []string) bool {
	for _, pat := range extra {
		if ok, _ := path.Match(pat, origin); ok || pat == "*" {
			return true
		}
	}
	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	if appSchemes[u.Scheme] {
		return true
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return false
	}
	if strings.EqualFold(u.Host, host) {
		return true
	}
	switch h := u.Hostname(); {
	case strings.EqualFold(h, "localhost"), strings.HasSuffix(strings.ToLower(h), ".localhost"), h == "0.0.0.0":
		return true
	default:
		ip := net.ParseIP(h)
		return ip != nil && ip.IsLoopback()
	}
}

// keyGiven reports the key sent as Authorization: Bearer or x-api-key.
func keyGiven(r *http.Request, key string) bool {
	got := r.Header.Get("x-api-key")
	if a := r.Header.Get("Authorization"); got == "" && len(a) > 7 && strings.EqualFold(a[:7], "Bearer ") {
		got = strings.TrimSpace(a[7:])
	}
	return got != "" && subtle.ConstantTimeCompare([]byte(got), []byte(key)) == 1
}

func splitList(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
