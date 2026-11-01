package link

import (
	"net"
	"testing"
)

func TestBridgeDevice(t *testing.T) {
	listing := `
Hardware Port: Ethernet
Device: en0
Ethernet Address: 1c:f6:4c:00:00:00

Hardware Port: Thunderbolt Bridge
Device: bridge0
Ethernet Address: 36:23:41:cf:a6:80

Hardware Port: Wi-Fi
Device: en1
`
	if got := bridgeDevice(listing); got != "bridge0" {
		t.Errorf("bridgeDevice = %q, want bridge0", got)
	}
	if got := bridgeDevice("Hardware Port: Wi-Fi\nDevice: en0\n"); got != "" {
		t.Errorf("bridgeDevice with no bridge = %q, want none", got)
	}
}

func TestParseAnswer(t *testing.T) {
	self := net.IPv4(169, 254, 217, 65)
	for _, c := range []struct {
		msg, want string
		ok        bool
	}{
		{"kinfer worker 1 169.254.154.219:50052", "169.254.154.219:50052", true},
		{"kinfer worker 1 169.254.217.65:50052", "", false}, // self
		{"kinfer worker 1 192.168.0.21:50052", "", false},   // not on a link-local cable
		{"kinfer worker 1 169.254.154.219", "", false},      // no port
		{"someone else 169.254.154.219:50052", "", false},
	} {
		got, ok := parseAnswer(c.msg, self)
		if got != c.want || ok != c.ok {
			t.Errorf("parseAnswer(%q) = %q, %v; want %q, %v", c.msg, got, ok, c.want, c.ok)
		}
	}
}

func TestCablePeers(t *testing.T) {
	out := `PING6(56=40+8+8 bytes) fe80::af:1691:32cc:1bb0%bridge0 --> ff02::1%bridge0
16 bytes from fe80::af:1691:32cc:1bb0%bridge0, icmp_seq=0 hlim=64 time=1.326 ms
16 bytes from fe80::1810:3c2f:eb6b:2480%bridge0, icmp_seq=0 hlim=64 time=2.237 ms
16 bytes from fe80::af:1691:32cc:1bb0%bridge0, icmp_seq=1 hlim=64 time=0.344 ms
16 bytes from fe80::1810:3c2f:eb6b:2480%bridge0, icmp_seq=1 hlim=64 time=0.410 ms
16 bytes from fe80::9:9:9:9%en0, icmp_seq=1 hlim=64 time=0.410 ms
`
	got := cablePeers(out, "bridge0", map[string]bool{"fe80::af:1691:32cc:1bb0": true})
	if len(got) != 1 || got[0] != "fe80::1810:3c2f:eb6b:2480%bridge0" {
		t.Errorf("cablePeers = %q", got)
	}
}
