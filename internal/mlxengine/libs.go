package mlxengine

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// The MLX libraries are not inside kinfer the way llama.cpp's are: with
// MLX's precompiled kernels they come to 239 MB (78 compressed), which every
// GGUF-only user would carry for nothing. kinfer fetches them the first time
// it serves an MLX model, checks them against the sha256 pinned here, and
// keeps them in ~/.kinfer/mlx/<version>; after that they are local, and a new
// MLX version simply lands in a directory of its own.
//
// The package is built by scripts/mlx-libs.sh: libmlxc.dylib (mlx-c at the
// commit for this MLX, rpath @loader_path), and libmlx.dylib, libjaccl.dylib
// and mlx.metallib from Apple's mlx-metal wheel, which need macOS 26.2.
var mlxLibs = struct {
	Version  string
	URL      string
	SHA256   string
	Size     int64
	MinMacOS [2]int
}{
	Version:  "0.32.4",
	URL:      "https://github.com/LocalKinAI/kinfer/releases/download/mlx-0.32.4/kinfer-mlx-0.32.4-darwin-arm64.tar.gz",
	SHA256:   "dfc2697d3ddc4b918de3902dbf39f9abb4a0894e2d7d6a8b2ad1e84e92de47e8",
	Size:     77645625,
	MinMacOS: [2]int{26, 2},
}

var libFiles = []string{"libmlxc.dylib", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib", "LICENSE-mlx", "LICENSE-mlx-c"}

// LibDir is where the MLX libraries are, or will be once fetched:
// $KINFER_MLX_LIB when set (a directory used as is, never fetched into), else
// ~/.kinfer/mlx/<version>.
func LibDir() string {
	if d := os.Getenv("KINFER_MLX_LIB"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".kinfer", "mlx", mlxLibs.Version)
}

// ensureLibs returns the library directory, fetching the libraries first if
// this machine has never had them.
func ensureLibs() (string, error) {
	dir := LibDir()
	if os.Getenv("KINFER_MLX_LIB") != "" || complete(dir) {
		return dir, nil
	}
	if err := Usable(); err != nil {
		return "", err
	}
	url := mlxLibs.URL
	if u := os.Getenv("KINFER_MLX_LIBS_URL"); u != "" {
		url = u
	}
	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		return "", err
	}

	log.Printf("mlx: fetching the MLX %s libraries (%.0f MB) — once; they are kept in %s",
		mlxLibs.Version, float64(mlxLibs.Size)/1e6, dir)
	tmp := fmt.Sprintf("%s.part-%d", dir, os.Getpid())
	defer os.RemoveAll(tmp)
	defer os.Remove(tmp + ".tar.gz")
	if err := fetch(url, tmp+".tar.gz"); err != nil {
		return "", fmt.Errorf("fetch MLX libraries from %s: %w", url, err)
	}
	if err := unpack(tmp+".tar.gz", tmp); err != nil {
		return "", fmt.Errorf("unpack MLX libraries: %w", err)
	}
	if err := os.WriteFile(filepath.Join(tmp, ".complete"), []byte(mlxLibs.SHA256+"\n"), 0o644); err != nil {
		return "", err
	}
	// Another kinfer may have finished the same fetch meanwhile; its copy is
	// as good as this one.
	if err := os.Rename(tmp, dir); err != nil {
		if complete(dir) {
			return dir, nil
		}
		os.RemoveAll(dir)
		if err := os.Rename(tmp, dir); err != nil {
			return "", err
		}
	}
	log.Printf("mlx: libraries ready in %s", dir)
	return dir, nil
}

// Usable says why this Mac cannot serve MLX models, or nil when it can: the
// libraries kinfer fetches need macOS 26.2.
func Usable() error {
	if major, minor, ok := macOS(); ok && (major < mlxLibs.MinMacOS[0] ||
		major == mlxLibs.MinMacOS[0] && minor < mlxLibs.MinMacOS[1]) {
		return fmt.Errorf("MLX models need macOS %d.%d or later; this Mac has %d.%d",
			mlxLibs.MinMacOS[0], mlxLibs.MinMacOS[1], major, minor)
	}
	return nil
}

// complete reports that dir holds every file of the pinned package.
func complete(dir string) bool {
	mark, err := os.ReadFile(filepath.Join(dir, ".complete"))
	if err != nil || strings.TrimSpace(string(mark)) != mlxLibs.SHA256 {
		return false
	}
	for _, f := range libFiles {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			return false
		}
	}
	return true
}

// fetch downloads url to path, checking size and sha256 as it goes.
func fetch(url, path string) error {
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
	var done int64
	next := time.Now().Add(5 * time.Second)
	buf := make([]byte, 1<<20)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := f.Write(buf[:n]); werr != nil {
				return werr
			}
			h.Write(buf[:n])
			done += int64(n)
			if time.Now().After(next) {
				log.Printf("mlx: %.0f of %.0f MB", float64(done)/1e6, float64(mlxLibs.Size)/1e6)
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
	if done != mlxLibs.Size {
		return fmt.Errorf("got %d bytes, want %d", done, mlxLibs.Size)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != mlxLibs.SHA256 {
		return fmt.Errorf("sha256 %s, want %s", got, mlxLibs.SHA256)
	}
	return f.Close()
}

// unpack extracts the package into dir, taking only the files it is known to
// hold: a name with a path in it, or one not on the list, is an error rather
// than something written somewhere.
func unpack(archive, dir string) error {
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	want := map[string]bool{}
	for _, n := range libFiles {
		want[n] = true
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
		if hd.Typeflag != tar.TypeReg || !want[hd.Name] {
			return fmt.Errorf("unexpected entry %q", hd.Name)
		}
		mode := os.FileMode(0o644)
		if strings.HasSuffix(hd.Name, ".dylib") {
			mode = 0o755
		}
		out, err := os.OpenFile(filepath.Join(dir, hd.Name), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
		if err != nil {
			return err
		}
		if _, err := io.Copy(out, tr); err != nil {
			out.Close()
			return err
		}
		if err := out.Close(); err != nil {
			return err
		}
		delete(want, hd.Name)
	}
	if len(want) > 0 {
		return fmt.Errorf("package is missing %d files", len(want))
	}
	return nil
}
