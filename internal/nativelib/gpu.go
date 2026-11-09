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
)

// The GPU backends off a Mac are not inside kinfer the way the CPU's are: most
// machines would carry them for nothing. The Vulkan backend is fetched the first
// time a model loads on a machine with a Vulkan driver, from llama.cpp's own
// release of the build embedded here, checked against the sha256 pinned below,
// and kept in ~/.kinfer/gpu. Only the backend's one file is taken from the
// archive: the rest of it is the same libraries kinfer already carries.
const vulkanRelease = "https://github.com/ggml-org/llama.cpp/releases/download/b11998/"

type vulkanAsset struct {
	Archive string
	SHA256  string
	Size    int64
	File    string // the backend's file name in the archive
}

var vulkanAssets = map[string]vulkanAsset{
	"linux/amd64":   {"llama-b11998-bin-ubuntu-vulkan-x64.tar.gz", "8f0b88dc3868cffa0d01d4ed16643117ecda4e036a3c041795d124424fc34c7e", 35492598, "libggml-vulkan.so"},
	"linux/arm64":   {"llama-b11998-bin-ubuntu-vulkan-arm64.tar.gz", "277440899b88e0adf95d2db52579ed813b68226688acde00802854aa99e5a8ae", 28864966, "libggml-vulkan.so"},
	"windows/amd64": {"llama-b11998-bin-win-vulkan-x64.zip", "47a32e3d164f8221658009d7f48794aed36ab9064791a5e04204a39d6eaf839a", 37363562, "ggml-vulkan.dll"},
	"windows/arm64": {"llama-b11998-bin-win-vulkan-arm64.zip", "6b177513efc87378f1d58636a4d05e265cc62812c33f676402328e04c977528a", 25565005, "ggml-vulkan.dll"},
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

// VulkanDir is where the backend is kept.
func VulkanDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".kinfer", "gpu", "vulkan-b11998")
}

// VulkanBackend returns the path of the Vulkan backend, fetching it the first
// time.
func VulkanBackend() (string, error) {
	asset, ok := vulkanAssets[runtime.GOOS+"/"+runtime.GOARCH]
	if !ok {
		return "", ErrNoVulkan
	}
	dir := VulkanDir()
	path := filepath.Join(dir, asset.File)
	if vulkanComplete(dir, path, asset) {
		return path, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}

	log.Printf("gpu: fetching the Vulkan backend (%.0f MB), once; it is kept in %s", float64(asset.Size)/1e6, dir)
	archive := filepath.Join(dir, fmt.Sprintf("%s.%d", asset.Archive, os.Getpid()))
	defer os.Remove(archive)
	if err := fetchVulkan(vulkanRelease+asset.Archive, archive, asset); err != nil {
		return "", fmt.Errorf("fetch %s: %w", asset.Archive, err)
	}
	tmp := fmt.Sprintf("%s.%d", path, os.Getpid())
	defer os.Remove(tmp)
	if err := extractMember(archive, asset.File, tmp); err != nil {
		return "", fmt.Errorf("unpack %s: %w", asset.Archive, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(dir, ".complete"), []byte(asset.SHA256+"\n"), 0o644); err != nil {
		return "", err
	}
	return path, nil
}

func vulkanComplete(dir, path string, asset vulkanAsset) bool {
	mark, err := os.ReadFile(filepath.Join(dir, ".complete"))
	if err != nil || strings.TrimSpace(string(mark)) != asset.SHA256 {
		return false
	}
	_, err = os.Stat(path)
	return err == nil
}

func fetchVulkan(url, path string, asset vulkanAsset) error {
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
	n, err := io.Copy(io.MultiWriter(f, h), resp.Body)
	if err != nil {
		return err
	}
	if n != asset.Size {
		return fmt.Errorf("got %d bytes, want %d", n, asset.Size)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != asset.SHA256 {
		return fmt.Errorf("sha256 %s, want %s", got, asset.SHA256)
	}
	return f.Close()
}

// extractMember writes the archive's regular file whose base name is name to
// dest, and nothing else: no name from inside the archive becomes a path here.
func extractMember(archive, name, dest string) error {
	if strings.HasSuffix(archive, ".zip") || strings.Contains(filepath.Base(archive), ".zip.") {
		z, err := zip.OpenReader(archive)
		if err != nil {
			return err
		}
		defer z.Close()
		for _, f := range z.File {
			if f.FileInfo().IsDir() || filepath.Base(filepath.ToSlash(f.Name)) != name {
				continue
			}
			r, err := f.Open()
			if err != nil {
				return err
			}
			defer r.Close()
			return writeExecutable(dest, r)
		}
		return fmt.Errorf("%s is not in the archive", name)
	}

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
			return fmt.Errorf("%s is not in the archive", name)
		}
		if err != nil {
			return err
		}
		if hd.Typeflag == tar.TypeReg && filepath.Base(hd.Name) == name {
			return writeExecutable(dest, tr)
		}
	}
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
