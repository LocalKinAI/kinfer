package nativelib

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// The GPU backends off a Mac are not inside kinfer the way the CPU's are: most
// machines would carry them for nothing. They come from llama.cpp's own release
// of the build embedded here, checked against the sha256 pinned below, and are
// kept in ~/.kinfer/gpu. Vulkan, about 35 MB, is fetched by itself the first time
// a model loads on a machine with a Vulkan driver. CUDA and ROCm run to hundreds
// of megabytes, so they wait to be asked for: kinfer gpu install.
//
// Only the files a backend needs are taken from an archive: the rest of it is
// the same libraries kinfer already carries.
const gpuRelease = "https://github.com/ggml-org/llama.cpp/releases/download/b11998/"

// Archive is one file of llama.cpp's release and what to keep from it.
type Archive struct {
	Name   string
	SHA256 string
	Size   int64
	Keep   func(base string) bool
}

// Backend is a GPU backend for this platform.
type Backend struct {
	Name     string // vulkan, cuda or rocm
	Variant  string // the CUDA or ROCm version, where there are several
	Lib      string // the file llama.cpp loads
	Archives []Archive
	Unpacked int64 // about what it takes on disk
}

func only(names ...string) func(string) bool {
	return func(base string) bool {
		for _, n := range names {
			if base == n {
				return true
			}
		}
		return false
	}
}

// cudaRuntime is what NVIDIA's runtime archive holds on Linux: libcudart and
// libcublas, and libcublasLt, which is most of its size and which libcublas needs.
func cudaRuntime(base string) bool { return strings.HasPrefix(base, "libcu") }

func dll(base string) bool { return strings.HasSuffix(strings.ToLower(base), ".dll") }

// rocmWindows keeps the HIP backend and the AMD runtime it ships with, and none
// of the llama.cpp libraries beside them, which kinfer has its own of.
func rocmWindows(base string) bool {
	n := strings.ToLower(base)
	if !strings.HasSuffix(n, ".dll") {
		return false
	}
	if n == "ggml-hip.dll" {
		return true
	}
	return !strings.HasPrefix(n, "ggml") && !strings.HasPrefix(n, "llama") && n != "mtmd.dll" && !strings.HasPrefix(n, "libomp")
}

// backendsFor lists a platform's backends, the one to prefer of each name first.
func backendsFor(platform string) []Backend {
	const so, win = "libggml-cuda.so", "ggml-cuda.dll"
	switch platform {
	case "linux/amd64":
		return []Backend{
			{"vulkan", "", "libggml-vulkan.so", []Archive{{"llama-b11998-bin-ubuntu-vulkan-x64.tar.gz", "8f0b88dc3868cffa0d01d4ed16643117ecda4e036a3c041795d124424fc34c7e", 35492598, only("libggml-vulkan.so")}}, 42e6},
			{"cuda", "13.4", so, []Archive{
				{"llama-b11998-bin-ubuntu-cuda-13.4-x64.tar.gz", "32276647fea6f193075bb22bf53e93c95489d2b37b7a97de961b699126dc5941", 163977046, only(so)},
				{"cudart-llama-b11998-bin-ubuntu-cuda-13.4-x64.tar.gz", "76399cdc44a60f312cd032dbbc5cd25c3546a59e8d79440fdfef7a4bff4e2f0c", 440232607, cudaRuntime},
			}, 775e6},
			{"cuda", "12.8", so, []Archive{
				{"llama-b11998-bin-ubuntu-cuda-12.8-x64.tar.gz", "fc28e673da01dbf9671ea98ea1e71b60dd8c56409e6500d5b0d180e9f3510626", 182566253, only(so)},
				{"cudart-llama-b11998-bin-ubuntu-cuda-12.8-x64.tar.gz", "16b7732eea008ea105c9161e954f19fbbdaa5ed4b094cce53c0d32ce794f4b48", 594371481, cudaRuntime},
			}, 0},
			{"rocm", "10.0", "libggml-hip.so", []Archive{{"llama-b11998-bin-ubuntu-rocm-10.0-x64.tar.gz", "8b2e8c572b5074fbb6c04721092a4c8536ec23a90cf8e251e3243f4d17f3c307", 252445199, only("libggml-hip.so")}}, 1192e6},
		}
	case "linux/arm64":
		return []Backend{
			{"vulkan", "", "libggml-vulkan.so", []Archive{{"llama-b11998-bin-ubuntu-vulkan-arm64.tar.gz", "277440899b88e0adf95d2db52579ed813b68226688acde00802854aa99e5a8ae", 28864966, only("libggml-vulkan.so")}}, 33e6},
			{"cuda", "13.4", so, []Archive{
				{"llama-b11998-bin-ubuntu-cuda-13.4-arm64.tar.gz", "5ac24d479d0fa3908de32a72c8b836e703000f3fad3e848a5c15daaa0c16ca03", 159098096, only(so)},
				{"cudart-llama-b11998-bin-ubuntu-cuda-13.4-arm64.tar.gz", "059d1f34cbcd7b6630f8977452dae676cd30da902c8570613c673de7fe07a00d", 552521413, cudaRuntime},
			}, 0},
		}
	case "windows/amd64":
		return []Backend{
			{"vulkan", "", "ggml-vulkan.dll", []Archive{{"llama-b11998-bin-win-vulkan-x64.zip", "47a32e3d164f8221658009d7f48794aed36ab9064791a5e04204a39d6eaf839a", 37363562, only("ggml-vulkan.dll")}}, 42e6},
			{"cuda", "13.4", win, []Archive{
				{"llama-b11998-bin-win-cuda-13.4-x64.zip", "ede53b39ce75578957d6ecba15a19dae723317313d39c051297aa73af13c4ce1", 165003467, only(win)},
				{"cudart-llama-bin-win-cuda-13.4-x64.zip", "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668", 423535356, dll},
			}, 703e6},
			{"cuda", "12.4", win, []Archive{
				{"llama-b11998-bin-win-cuda-12.4-x64.zip", "a52b4f9de136e6fe245e2aa0c35b36d6886f79de3f71bfe04278b11b3113f49b", 279781938, only(win)},
				{"cudart-llama-bin-win-cuda-12.4-x64.zip", "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6", 391443627, dll},
			}, 0},
			{"rocm", "10.0", "ggml-hip.dll", []Archive{{"llama-b11998-bin-win-rocm-10.0-x64.zip", "48607a777abe986076a59d1e1d8d663ae88ee37e0894c92e4656915f05586cac", 265551520, rocmWindows}}, 1200e6},
		}
	case "windows/arm64":
		return []Backend{
			{"vulkan", "", "ggml-vulkan.dll", []Archive{{"llama-b11998-bin-win-vulkan-arm64.zip", "6b177513efc87378f1d58636a4d05e265cc62812c33f676402328e04c977528a", 25565005, only("ggml-vulkan.dll")}}, 30e6},
			{"cuda", "13.4", win, []Archive{
				{"llama-b11998-bin-win-cuda-13.4-arm64.zip", "1ee94460cf6e0f51cb54339b783f51ce01560f15e7b2c9eb0c66db7e6b609f1f", 152762510, only(win)},
				{"cudart-llama-bin-win-cuda-13.4-arm64.zip", "642dcde8805b3e3165ca710a5443b3b4044b27d96bd3ee3132473988c9bcb774", 153262407, dll},
			}, 0},
		}
	}
	return nil
}

// Backends lists this platform's GPU backends.
func Backends() []Backend { return backendsFor(runtime.GOOS + "/" + runtime.GOARCH) }

// FindBackend finds a backend by name, and by version when one is given.
func FindBackend(name, variant string) (Backend, bool) {
	for _, b := range Backends() {
		if b.Name == name && (variant == "" || b.Variant == variant) {
			return b, true
		}
	}
	return Backend{}, false
}

// ErrNoVulkan says this platform has no Vulkan backend to fetch.
var ErrNoVulkan = errors.New("no Vulkan backend for this platform")

// VulkanLoader is the name of the system's Vulkan loader, which a graphics
// driver brings with it: a machine without one has nothing for the backend to
// talk to, and the backend is not fetched.
func VulkanLoader() string {
	if runtime.GOOS == "windows" {
		return "vulkan-1.dll"
	}
	return "libvulkan.so.1"
}

// VulkanBackend returns the path of the Vulkan backend, fetching it the first
// time.
func VulkanBackend() (string, error) {
	b, ok := FindBackend("vulkan", "")
	if !ok {
		return "", ErrNoVulkan
	}
	if err := b.Install(); err != nil {
		return "", err
	}
	return b.Path(), nil
}

func gpuRoot() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".kinfer", "gpu")
}

// ID names a backend and its version: cuda-13.4.
func (b Backend) ID() string {
	if b.Variant == "" {
		return b.Name
	}
	return b.Name + "-" + b.Variant
}

// Dir is where the backend is kept.
func (b Backend) Dir() string { return filepath.Join(gpuRoot(), b.ID()+"-b11998") }

// Path is the library llama.cpp loads.
func (b Backend) Path() string { return filepath.Join(b.Dir(), b.Lib) }

// Download is how much fetching it costs.
func (b Backend) Download() int64 {
	var n int64
	for _, a := range b.Archives {
		n += a.Size
	}
	return n
}

func (b Backend) stamp() string {
	var s []string
	for _, a := range b.Archives {
		s = append(s, a.SHA256)
	}
	return strings.Join(s, "\n") + "\n"
}

// Installed reports that the backend has been fetched whole.
func (b Backend) Installed() bool {
	mark, err := os.ReadFile(filepath.Join(b.Dir(), ".complete"))
	if err != nil || string(mark) != b.stamp() {
		return false
	}
	_, err = os.Stat(b.Path())
	return err == nil
}

// Install fetches the backend unless it is already here. Nothing lands under
// its name until all of it has arrived.
func (b Backend) Install() error {
	if b.Installed() {
		return nil
	}
	if err := os.MkdirAll(gpuRoot(), 0o755); err != nil {
		return err
	}
	log.Printf("gpu: fetching the %s backend (%.0f MB), once; it is kept in %s", b.ID(), float64(b.Download())/1e6, b.Dir())
	tmp := fmt.Sprintf("%s.part-%d", b.Dir(), os.Getpid())
	if err := os.RemoveAll(tmp); err != nil {
		return err
	}
	if err := os.MkdirAll(tmp, 0o755); err != nil {
		return err
	}
	defer os.RemoveAll(tmp)

	for _, a := range b.Archives {
		// The archive's own name, with its extension, says how to read it.
		archive := filepath.Join(tmp, ".archive-"+a.Name)
		if err := fetchArchive(gpuRelease+a.Name, archive, a); err != nil {
			return fmt.Errorf("fetch %s: %w", a.Name, err)
		}
		if err := extractMatching(archive, a.Keep, tmp); err != nil {
			return fmt.Errorf("unpack %s: %w", a.Name, err)
		}
		os.Remove(archive)
	}
	if _, err := os.Stat(filepath.Join(tmp, b.Lib)); err != nil {
		return fmt.Errorf("%s was not in the release's archives", b.Lib)
	}
	if err := os.WriteFile(filepath.Join(tmp, ".complete"), []byte(b.stamp()), 0o644); err != nil {
		return err
	}
	if err := os.RemoveAll(b.Dir()); err != nil {
		return err
	}
	return os.Rename(tmp, b.Dir())
}

// Remove deletes the backend, and stops using it if it was in use.
func (b Backend) Remove() error {
	if a, ok := ActiveBackend(); ok && a.ID() == b.ID() {
		if err := ClearActiveBackend(); err != nil {
			return err
		}
	}
	return os.RemoveAll(b.Dir())
}

func activeFile() string { return filepath.Join(gpuRoot(), "active") }

// ActiveBackend is the backend the user chose, if any. With none, Vulkan is used
// where there is a driver for it.
func ActiveBackend() (Backend, bool) {
	raw, err := os.ReadFile(activeFile())
	if err != nil {
		return Backend{}, false
	}
	id := strings.TrimSpace(string(raw))
	for _, b := range Backends() {
		if b.ID() == id {
			return b, true
		}
	}
	return Backend{}, false
}

// SetActiveBackend makes b the backend models load on.
func SetActiveBackend(b Backend) error {
	if err := os.MkdirAll(gpuRoot(), 0o755); err != nil {
		return err
	}
	return os.WriteFile(activeFile(), []byte(b.ID()+"\n"), 0o644)
}

// ClearActiveBackend goes back to Vulkan where there is a driver for it.
func ClearActiveBackend() error {
	if err := os.Remove(activeFile()); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func fetchArchive(url, path string, a Archive) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s", resp.Status)
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	buf := make([]byte, 1<<20)
	var done int64
	next := time.Now().Add(5 * time.Second)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := f.Write(buf[:n]); werr != nil {
				return werr
			}
			h.Write(buf[:n])
			done += int64(n)
			if time.Now().After(next) {
				log.Printf("gpu: %.0f of %.0f MB", float64(done)/1e6, float64(a.Size)/1e6)
				next = time.Now().Add(5 * time.Second)
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}
	if done != a.Size {
		return fmt.Errorf("got %d bytes, want %d", done, a.Size)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != a.SHA256 {
		return fmt.Errorf("sha256 %s, want %s", got, a.SHA256)
	}
	return f.Close()
}

// extractMatching writes the archive's regular files whose base name keep
// accepts into dir, flat, under that base name: no path from inside the archive
// becomes a path here. It is an error if nothing matches.
func extractMatching(archive string, keep func(string) bool, dir string) error {
	found := 0
	if strings.HasSuffix(archive, ".zip") {
		z, err := zip.OpenReader(archive)
		if err != nil {
			return err
		}
		defer z.Close()
		for _, f := range z.File {
			base := filepath.Base(filepath.ToSlash(f.Name))
			if f.FileInfo().IsDir() || !keep(base) {
				continue
			}
			r, err := f.Open()
			if err != nil {
				return err
			}
			err = writeExecutable(filepath.Join(dir, base), r)
			r.Close()
			if err != nil {
				return err
			}
			found++
		}
	} else {
		f, err := os.Open(archive)
		if err != nil {
			return err
		}
		defer f.Close()
		gz, err := gzip.NewReader(f)
		if err != nil {
			return err
		}
		tr := tar.NewReader(gz)
		for {
			hd, err := tr.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				return err
			}
			base := filepath.Base(hd.Name)
			if hd.Typeflag != tar.TypeReg || !keep(base) {
				continue
			}
			if err := writeExecutable(filepath.Join(dir, base), tr); err != nil {
				return err
			}
			found++
		}
	}
	if found == 0 {
		return errors.New("nothing to take from it")
	}
	return nil
}

func writeExecutable(dest string, r io.Reader) error {
	out, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, r); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
