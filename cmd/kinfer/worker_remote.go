package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/link"
)

// remoteWorker starts `kinfer worker` on another Mac over ssh, detached there
// so it outlives the session, or stops the one running there — so the Mac
// that wants the GPU can lend it to itself without anyone at the other.
//
// ssh asks for whatever it needs (a password, a new host key) on this
// terminal; with a key in the other Mac's authorized_keys it asks nothing.
// The worker started this way lasts until that Mac restarts.
//
// on is user@host, or a user alone for the Mac at the other end of the
// Thunderbolt cable (cableHost).
func remoteWorker(on string, stop bool) error {
	host, err := cableHost(on)
	if err != nil {
		return err
	}
	args := []string{"-o", "ConnectTimeout=10"}
	if host != on {
		// The Mac at the other end of a cable is taken at its word the first
		// time, its host key kept; a key that changes later is still refused.
		args = append(args, "-o", "StrictHostKeyChecking=accept-new")
	}
	script := remoteStartScript
	if stop {
		script = remoteStopScript
	}
	// ssh asks on the terminal itself; only what the script prints is read.
	var out bytes.Buffer
	cmd := exec.Command("ssh", append(args, host, script)...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, &out, os.Stderr
	err = cmd.Run()
	there := ""
	for _, l := range strings.Split(strings.TrimRight(out.String(), "\n"), "\n") {
		if v, ok := strings.CutPrefix(l, "kinfer-version: "); ok {
			there = v
			continue
		}
		if l != "" {
			fmt.Println(l)
		}
	}
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok && ee.ExitCode() == 3 {
			return fmt.Errorf("no kinfer on %s: install it there first (the curl installer or brew, see the README)", host)
		}
		return fmt.Errorf("ssh %s: %w", host, err)
	}
	// The protocol is checked when a model is borrowed; a worker of another
	// release is refused then, so say it now, while it is one command to fix.
	if here := "kinfer " + version; there != "" && there != here {
		fmt.Printf("but %s runs %s and this Mac %s: a worker is borrowed only by the same release, so upgrade the older one\n",
			host, there, here)
	}
	return nil
}

// cableHost is on as ssh takes it: user@host as given, or a user alone at the
// Mac on the other end of the Thunderbolt cable, by its link-local address.
func cableHost(on string) (string, error) {
	if strings.Contains(on, "@") {
		return on, nil
	}
	b, err := link.FindBridge()
	if err != nil {
		return "", fmt.Errorf("%w; give -on user@host", err)
	}
	peers, err := link.CablePeers(b)
	switch {
	case err != nil:
		return "", fmt.Errorf("looking for the Mac at the other end of the Thunderbolt cable: %w; give -on user@host", err)
	case len(peers) == 0:
		return "", fmt.Errorf("no Mac answers at the other end of the Thunderbolt cable (%s); give -on user@host", b.Name)
	case len(peers) > 1:
		return "", fmt.Errorf("%d Macs answer on the Thunderbolt cable (%s); give -on %s@ and one of them", len(peers), strings.Join(peers, ", "), on)
	}
	fmt.Printf("the Mac at the other end of the Thunderbolt cable: %s\n", peers[0])
	return on + "@" + peers[0], nil
}

// remoteFindKinfer finds kinfer on the other Mac: a shell ssh runs reads no
// profile, so ~/.local/bin and Homebrew are looked in by hand.
const remoteFindKinfer = `k=$(command -v kinfer 2>/dev/null)
for c in "$k" "$HOME/.local/bin/kinfer" /opt/homebrew/bin/kinfer /usr/local/bin/kinfer; do
  if [ -n "$c" ] && [ -x "$c" ]; then k=$c; break; fi
done
[ -n "$k" ] && [ -x "$k" ] || exit 3
`

const remoteStartScript = remoteFindKinfer + `echo "kinfer-version: $("$k" version | cut -d, -f1)"
host=$(scutil --get LocalHostName 2>/dev/null || hostname)
if pgrep -f 'kinfer[^ ]* worker' >/dev/null; then
  echo "a kinfer worker already runs on $host:"
  ps -o pid=,command= -p "$(pgrep -f 'kinfer[^ ]* worker' | head -1)"
  exit 0
fi
mkdir -p "$HOME/.kinfer"
log="$HOME/.kinfer/worker.log"
nohup "$k" worker > "$log" 2>&1 < /dev/null &
pid=$!
i=0
while [ $i -lt 60 ] && kill -0 "$pid" 2>/dev/null && ! grep -q '^kinfer worker on' "$log"; do
  sleep 0.5; i=$((i + 1))
done
if kill -0 "$pid" 2>/dev/null; then
  echo "started a kinfer worker on $host (pid $pid, log $log):"
  grep -E '^kinfer worker on|^  ' "$log" | head -6
else
  echo "the kinfer worker on $host did not start:"
  grep -vE '^(ggml|llama|load|print_info)' "$log" | tail -5
  exit 1
fi
`

const remoteStopScript = `host=$(scutil --get LocalHostName 2>/dev/null || hostname)
if pkill -f 'kinfer[^ ]* worker'; then
  echo "stopped the kinfer worker on $host"
else
  echo "no kinfer worker runs on $host"
fi
`
