package link

import (
	"context"
	"net"
	"os/exec"
	"slices"
	"strconv"
	"syscall"
	"time"
)

// FindBridge returns this machine's end of the Thunderbolt cable, or
// ErrNoBridge when there is no cable or it has no address yet.
func FindBridge() (Bridge, error) {
	name := "bridge0" // what macOS names it, if networksetup cannot say
	if out, err := exec.Command("networksetup", "-listallhardwareports").Output(); err == nil {
		if dev := bridgeDevice(string(out)); dev != "" {
			name = dev
		}
	}
	ifi, err := net.InterfaceByName(name)
	if err != nil || ifi.Flags&net.FlagUp == 0 {
		return Bridge{}, ErrNoBridge
	}
	addrs, err := ifi.Addrs()
	if err != nil {
		return Bridge{}, ErrNoBridge
	}
	for _, a := range addrs {
		if n, ok := a.(*net.IPNet); ok && n.IP.To4() != nil && linkLocal.Contains(n.IP) {
			return Bridge{Name: name, Index: ifi.Index, IP: n.IP.To4()}, nil
		}
	}
	return Bridge{}, ErrNoBridge
}

// boundTo keeps a socket to one interface (IP_BOUND_IF): what it sends leaves
// there, and it hears only what arrives there.
func boundTo(index int) func(network, address string, c syscall.RawConn) error {
	return func(network, address string, c syscall.RawConn) error {
		var serr error
		err := c.Control(func(fd uintptr) {
			serr = syscall.SetsockoptInt(int(fd), syscall.IPPROTO_IP, syscall.IP_BOUND_IF, index)
		})
		if err != nil {
			return err
		}
		return serr
	}
}

// Answer tells whoever asks on b that a worker listens at addr, until ctx ends.
// The question is a broadcast to the bridge's link-local network, heard on the
// bridge alone; the answer goes back the same way.
func Answer(ctx context.Context, b Bridge, addr string) error {
	lc := net.ListenConfig{Control: boundTo(b.Index)}
	pc, err := lc.ListenPacket(ctx, "udp4", ":"+strconv.Itoa(Port))
	if err != nil {
		return err
	}
	go func() {
		<-ctx.Done()
		pc.Close()
	}()
	buf := make([]byte, 512)
	for {
		n, from, err := pc.ReadFrom(buf)
		if err != nil {
			return ctx.Err()
		}
		u, ok := from.(*net.UDPAddr)
		if !ok || !linkLocal.Contains(u.IP) || string(buf[:n]) != ask {
			continue
		}
		_, _ = pc.WriteTo([]byte(answer+addr), from)
	}
}

// Find asks on b for kinfer workers and returns the addresses that answer
// within wait, sorted.
func Find(b Bridge, wait time.Duration) ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), wait)
	defer cancel()
	lc := net.ListenConfig{Control: boundTo(b.Index)}
	pc, err := lc.ListenPacket(ctx, "udp4", net.JoinHostPort(b.IP.String(), "0"))
	if err != nil {
		return nil, err
	}
	// Closing it took a second on the MacBook Pro, every time, after the
	// answers were in; nothing waits for that.
	defer func() { go pc.Close() }()
	// The link-local network's own broadcast address: macOS keeps a route for
	// it scoped to the bridge, which the socket is bound to. Neither the
	// limited broadcast (255.255.255.255) nor a multicast group has one there,
	// and those datagrams never left the bridge.
	to := &net.UDPAddr{IP: net.IPv4(169, 254, 255, 255), Port: Port}
	// Asked twice, a third of the way apart, in case the first datagram is lost.
	if _, err := pc.WriteTo([]byte(ask), to); err != nil {
		return nil, err
	}
	again := time.AfterFunc(wait/3, func() { _, _ = pc.WriteTo([]byte(ask), to) })
	defer again.Stop()

	_ = pc.SetReadDeadline(time.Now().Add(wait))
	var found []string
	buf := make([]byte, 512)
	for {
		n, _, err := pc.ReadFrom(buf)
		if err != nil {
			break // the deadline
		}
		if addr, ok := parseAnswer(string(buf[:n]), b.IP); ok && !slices.Contains(found, addr) {
			found = append(found, addr)
		}
	}
	slices.Sort(found)
	return found, nil
}

// CablePeers is the other Macs on b, as the IPv6 link-local addresses ssh
// takes ("fe80::…%bridge0"). Every Mac answers the all-nodes multicast, so
// nothing of kinfer's needs to run at the other end to be found.
func CablePeers(b Bridge) ([]string, error) {
	out, err := exec.Command("ping6", "-c", "2", "-i", "0.3", "ff02::1%"+b.Name).Output()
	if err != nil && len(out) == 0 {
		return nil, err
	}
	own := map[string]bool{}
	if ifi, err := net.InterfaceByIndex(b.Index); err == nil {
		if addrs, err := ifi.Addrs(); err == nil {
			for _, a := range addrs {
				if n, ok := a.(*net.IPNet); ok {
					own[n.IP.String()] = true
				}
			}
		}
	}
	return cablePeers(string(out), b.Name, own), nil
}
