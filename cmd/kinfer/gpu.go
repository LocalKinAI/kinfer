package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/dl"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
)

// kinfer gpu shows and changes which GPU backend models load on, off a Mac.
// Vulkan arrives by itself. CUDA and ROCm are hundreds of megabytes, so they are
// fetched only when asked for, after saying how much.

const gpuUsage = `usage:
  kinfer gpu                            what is installed, and in use
  kinfer gpu install cuda|rocm|vulkan   fetch a backend; it says the size and asks first
      -y          do not ask
      -version V  a particular CUDA, such as 12.8 (default: what the driver can run)
  kinfer gpu use cuda|rocm|vulkan       load models on a backend already fetched
  kinfer gpu remove cuda|rocm|vulkan    delete a backend
`

func cmdGPU(args []string) error {
	if runtime.GOOS == "darwin" {
		fmt.Println("this Mac's GPU is used through Metal, which is built in; there is nothing to install")
		return nil
	}
	if len(args) == 0 {
		return gpuStatus()
	}
	switch args[0] {
	case "status":
		return gpuStatus()
	case "install":
		return gpuInstall(args[1:])
	case "use":
		return gpuUse(args[1:])
	case "remove", "rm":
		return gpuRemove(args[1:])
	}
	return errors.New(gpuUsage)
}

func gpuStatus() error {
	active, chosen := nativelib.ActiveBackend()
	if strings.EqualFold(os.Getenv("KINFER_GPU"), "off") {
		fmt.Println("KINFER_GPU=off: models run on the CPU")
	}
	if v, ok := driverCUDA(); ok {
		fmt.Printf("NVIDIA driver: runs CUDA %d\n", v)
	}
	if _, err := dl.Open(nativelib.VulkanLoader()); err == nil {
		fmt.Println("Vulkan driver: found")
	}
	fmt.Printf("\n%-12s %10s  %s\n", "BACKEND", "DOWNLOAD", "")
	for _, b := range nativelib.Backends() {
		state := "not fetched"
		if b.Installed() {
			state = "fetched"
		}
		if chosen && b.ID() == active.ID() {
			state = "in use"
		} else if !chosen && b.Name == "vulkan" && b.Installed() {
			state = "in use, where there is a driver"
		}
		fmt.Printf("%-12s %7.0f MB  %s\n", b.ID(), float64(b.Download())/1e6, state)
	}
	return nil
}

// driverCUDA is the newest CUDA the NVIDIA driver here can run.
func driverCUDA() (int, bool) {
	out, err := exec.Command("nvidia-smi").Output()
	if err != nil {
		return 0, false
	}
	return parseDriverCUDA(string(out))
}

var cudaVersion = regexp.MustCompile(`CUDA Version:\s*(\d+)\.`)

func parseDriverCUDA(s string) (int, bool) {
	m := cudaVersion.FindStringSubmatch(s)
	if m == nil {
		return 0, false
	}
	v, err := strconv.Atoi(m[1])
	return v, err == nil
}

// pickCUDA is the newest of the CUDA builds the driver can run.
func pickCUDA(driver int) (nativelib.Backend, error) {
	for _, b := range nativelib.Backends() {
		if b.Name != "cuda" {
			continue
		}
		major, _ := strconv.Atoi(strings.SplitN(b.Variant, ".", 2)[0])
		if major <= driver {
			return b, nil
		}
	}
	return nativelib.Backend{}, fmt.Errorf("the NVIDIA driver here runs CUDA %d, older than any build on offer", driver)
}

func gpuInstall(args []string) error {
	fs := flag.NewFlagSet("gpu install", flag.ExitOnError)
	yes := fs.Bool("y", false, "do not ask")
	version := fs.String("version", "", "a particular version of the backend")
	force := fs.Bool("force", false, "fetch it even with no driver for it here")
	name := ""
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		name, args = args[0], args[1:]
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if name == "" {
		name = fs.Arg(0)
	}

	b, err := chooseForInstall(name, *version, *force)
	if err != nil {
		return err
	}

	mb := float64(b.Download()) / 1e6
	if b.Unpacked > 0 {
		fmt.Printf("%s downloads %.0f MB and takes about %.1f GB on disk.\n", b.ID(), mb, float64(b.Unpacked)/1e9)
	} else {
		fmt.Printf("%s downloads %.0f MB.\n", b.ID(), mb)
	}
	if !*yes && !b.Installed() {
		fmt.Print("fetch it? [y/N] ")
		line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
		if a := strings.ToLower(strings.TrimSpace(line)); a != "y" && a != "yes" {
			return errors.New("not fetched")
		}
	}
	if err := b.Install(); err != nil {
		return err
	}

	p := engine.ProbeBackend(b.Path())
	if !p.OK {
		return fmt.Errorf("%s is fetched but did not start here (%s), so models stay where they were; kinfer gpu remove %s deletes it", b.ID(), p.Error, b.Name)
	}
	if len(p.GPUs()) == 0 {
		return fmt.Errorf("%s is fetched but found no GPU, so models stay where they were; kinfer gpu remove %s deletes it", b.ID(), b.Name)
	}
	if err := nativelib.SetActiveBackend(b); err != nil {
		return err
	}
	fmt.Printf("models now load on %s (%s), experimental. Try it: kinfer selftest\n", b.ID(), p.GPUs()[0].Name)
	fmt.Println("a daemon or server already running keeps what it had until it restarts")
	return nil
}

func chooseForInstall(name, version string, force bool) (nativelib.Backend, error) {
	switch name {
	case "cuda":
		if version == "" && !force {
			driver, ok := driverCUDA()
			if !ok {
				return nativelib.Backend{}, errors.New("no NVIDIA driver found here (nvidia-smi did not answer); CUDA needs one")
			}
			return pickCUDA(driver)
		}
	case "rocm":
		if !force && runtime.GOOS == "linux" {
			if _, err := dl.Open("libamdhip64.so.7"); err != nil {
				return nativelib.Backend{}, errors.New("AMD's ROCm 10 runtime was not found here, and the Linux backend needs it installed: https://rocm.docs.amd.com")
			}
		}
	case "vulkan":
	default:
		return nativelib.Backend{}, errors.New(gpuUsage)
	}
	b, ok := nativelib.FindBackend(name, version)
	if !ok {
		return nativelib.Backend{}, fmt.Errorf("no %s %s for %s/%s", name, version, runtime.GOOS, runtime.GOARCH)
	}
	return b, nil
}

// fetched finds a backend already here by name.
func fetched(args []string) (nativelib.Backend, error) {
	if len(args) != 1 {
		return nativelib.Backend{}, errors.New(gpuUsage)
	}
	for _, b := range nativelib.Backends() {
		if (b.Name == args[0] || b.ID() == args[0]) && b.Installed() {
			return b, nil
		}
	}
	return nativelib.Backend{}, fmt.Errorf("%s has not been fetched; kinfer gpu install %s", args[0], args[0])
}

func gpuUse(args []string) error {
	b, err := fetched(args)
	if err != nil {
		return err
	}
	p := engine.ProbeBackend(b.Path())
	if !p.OK || len(p.GPUs()) == 0 {
		return fmt.Errorf("%s did not start here or found no GPU (%s), so models stay where they were", b.ID(), p.Error)
	}
	if err := nativelib.SetActiveBackend(b); err != nil {
		return err
	}
	fmt.Printf("models now load on %s (%s)\n", b.ID(), p.GPUs()[0].Name)
	return nil
}

func gpuRemove(args []string) error {
	b, err := fetched(args)
	if err != nil {
		return err
	}
	if err := b.Remove(); err != nil {
		return err
	}
	fmt.Printf("removed %s\n", b.ID())
	return nil
}
