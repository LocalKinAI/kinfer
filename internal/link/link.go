// Package link finds the Thunderbolt cable between two Macs, and the kinfer
// workers at its other end.
//
// A worker lends its GPU over ggml's RPC protocol, which has no authentication,
// so it belongs on the cable between two machines and nowhere else. On a Mac
// that cable is the Thunderbolt Bridge, an interface with a link-local address
// that changes whenever the cable is plugged in again. Rather than have anyone
// look that address up, a worker answers a question broadcast on the bridge, and
// a machine that wants a GPU asks: one datagram out, one back from each worker,
// on that interface only — never Wi-Fi, which takes a link-local address too
// when there is no DHCP.
package link

import (
	"errors"
	"net"
	"slices"
	"strconv"
	"strings"
)

// Port is a worker's port: TCP for the work, UDP for being found.
const Port = 50052

// ErrNoBridge is the answer on a machine with no Thunderbolt cable to another.
var ErrNoBridge = errors.New("no Thunderbolt Bridge with an address here: no Thunderbolt cable to another Mac")

// Bridge is this machine's end of the Thunderbolt cable.
type Bridge struct {
	Name  string // bridge0
	Index int
	IP    net.IP // link-local, 169.254.x.x
}

// Addr is where a worker on this bridge listens.
func (b Bridge) Addr() string { return net.JoinHostPort(b.IP.String(), strconv.Itoa(Port)) }

const (
	ask    = "kinfer find 1"
	answer = "kinfer worker 1 "
)

var linkLocal = &net.IPNet{IP: net.IPv4(169, 254, 0, 0), Mask: net.CIDRMask(16, 32)}

// bridgeDevice picks the Thunderbolt Bridge's device out of
// `networksetup -listallhardwareports`; "" when it lists none.
func bridgeDevice(listing string) string {
	lines := strings.Split(listing, "\n")
	for i, l := range lines {
		if strings.TrimSpace(l) != "Hardware Port: Thunderbolt Bridge" || i+1 >= len(lines) {
			continue
		}
		if dev, ok := strings.CutPrefix(strings.TrimSpace(lines[i+1]), "Device: "); ok {
			return strings.TrimSpace(dev)
		}
	}
	return ""
}

// parseAnswer reads a worker's answer: its address, if it is one on the
// link-local network other than self.
func parseAnswer(msg string, self net.IP) (string, bool) {
	addr, ok := strings.CutPrefix(msg, answer)
	if !ok {
		return "", false
	}
	host, port, err := net.SplitHostPort(strings.TrimSpace(addr))
	if err != nil {
		return "", false
	}
	ip := net.ParseIP(host)
	if ip == nil || !linkLocal.Contains(ip) || ip.Equal(self) {
		return "", false
	}
	return net.JoinHostPort(ip.String(), port), true
}

// cablePeers reads ping6's replies from the all-nodes multicast on bridge:
// each other Mac's link-local address, as ssh takes it, sorted.
func cablePeers(out, bridge string, own map[string]bool) []string {
	var peers []string
	for _, l := range strings.Split(out, "\n") {
		_, rest, ok := strings.Cut(l, " bytes from ")
		if !ok {
			continue
		}
		addr, _, _ := strings.Cut(rest, ",")
		ip, zone, _ := strings.Cut(strings.TrimSuffix(addr, ":"), "%")
		if zone != bridge || own[ip] || !strings.HasPrefix(ip, "fe80:") {
			continue
		}
		if p := ip + "%" + zone; !slices.Contains(peers, p) {
			peers = append(peers, p)
		}
	}
	slices.Sort(peers)
	return peers
}
