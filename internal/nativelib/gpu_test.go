package nativelib

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"os"
	"path/filepath"
	"testing"
)

func writeTGZ(t *testing.T, path string, files map[string]string) {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for name, body := range files {
		tw.WriteHeader(&tar.Header{Name: name, Mode: 0o755, Size: int64(len(body)), Typeflag: tar.TypeReg})
		tw.Write([]byte(body))
	}
	tw.Close()
	gz.Close()
	os.WriteFile(path, buf.Bytes(), 0o644)
}

func writeZip(t *testing.T, path string, files map[string]string) {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for name, body := range files {
		w, _ := zw.Create(name)
		w.Write([]byte(body))
	}
	zw.Close()
	os.WriteFile(path, buf.Bytes(), 0o644)
}

func TestExtractMatchingTakesOnlyWhatItIsAsked(t *testing.T) {
	dir := t.TempDir()
	tgz := filepath.Join(dir, "a.tar.gz")
	writeTGZ(t, tgz, map[string]string{"llama/other.so": "no", "llama/libggml-vulkan.so": "backend", "../evil/libcublas.so.13": "rt"})
	zp := filepath.Join(dir, "a.zip")
	writeZip(t, zp, map[string]string{"other.txt": "no", "sub/ggml-cuda.dll": "backend"})

	out := filepath.Join(dir, "out")
	os.Mkdir(out, 0o755)
	if err := extractMatching(tgz, only("libggml-vulkan.so"), out); err != nil {
		t.Fatal(err)
	}
	if err := extractMatching(tgz, cudaRuntime, out); err != nil {
		t.Fatal(err)
	}
	if err := extractMatching(zp, dll, out); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"libggml-vulkan.so", "libcublas.so.13", "ggml-cuda.dll"} {
		if _, err := os.Stat(filepath.Join(out, name)); err != nil {
			t.Errorf("%s was not extracted flat: %v", name, err)
		}
	}
	if _, err := os.Stat(filepath.Join(out, "other.so")); err == nil {
		t.Error("a file nobody asked for was extracted")
	}
	if _, err := os.Stat(filepath.Join(dir, "evil")); err == nil {
		t.Error("a path inside the archive became a path here")
	}
	if err := extractMatching(zp, only("absent"), out); err == nil {
		t.Error("an archive with nothing to take reported no error")
	}
}

func TestRocmWindowsKeepsTheRuntimeNotTheSiblings(t *testing.T) {
	for name, want := range map[string]bool{
		"ggml-hip.dll": true, "amdhip64_7.dll": true, "hipblas.dll": true, "amd_comgr.dll": true,
		"llama.dll": false, "ggml.dll": false, "ggml-base.dll": false, "ggml-cpu-zen4.dll": false,
		"mtmd.dll": false, "libomp.dll": false, "llama-server-impl.dll": false, "readme.txt": false,
	} {
		if got := rocmWindows(name); got != want {
			t.Errorf("rocmWindows(%s) = %v, want %v", name, got, want)
		}
	}
}

func TestEveryPlatformPinsItsArchives(t *testing.T) {
	for _, p := range []string{"linux/amd64", "linux/arm64", "windows/amd64", "windows/arm64"} {
		list := backendsFor(p)
		if len(list) == 0 || list[0].Name != "vulkan" {
			t.Errorf("%s: Vulkan is not first in %+v", p, list)
		}
		for _, b := range list {
			if b.Lib == "" || len(b.Archives) == 0 {
				t.Errorf("%s %s is incomplete", p, b.ID())
			}
			for _, a := range b.Archives {
				if len(a.SHA256) != 64 || a.Size == 0 || a.Keep == nil {
					t.Errorf("%s %s %s is not pinned", p, b.ID(), a.Name)
				}
			}
		}
	}
	if len(backendsFor("darwin/arm64")) != 0 {
		t.Error("a Mac has Metal built in, and no backend to fetch")
	}
}

func TestActiveBackendRoundTrips(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	bs := Backends()
	if len(bs) == 0 {
		t.Skip("no backends on this platform")
	}
	if _, ok := ActiveBackend(); ok {
		t.Fatal("something is active before anything was chosen")
	}
	if err := SetActiveBackend(bs[0]); err != nil {
		t.Fatal(err)
	}
	if a, ok := ActiveBackend(); !ok || a.ID() != bs[0].ID() {
		t.Errorf("active = %+v, %v", a, ok)
	}
	if err := bs[0].Remove(); err != nil {
		t.Fatal(err)
	}
	if _, ok := ActiveBackend(); ok {
		t.Error("the backend removed is still active")
	}
}
