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

func TestExtractMemberTakesOnlyTheNamedFile(t *testing.T) {
	dir := t.TempDir()

	var tgz bytes.Buffer
	gz := gzip.NewWriter(&tgz)
	tw := tar.NewWriter(gz)
	for name, body := range map[string]string{"llama/other.so": "no", "llama/libggml-vulkan.so": "backend"} {
		tw.WriteHeader(&tar.Header{Name: name, Mode: 0o755, Size: int64(len(body)), Typeflag: tar.TypeReg})
		tw.Write([]byte(body))
	}
	tw.Close()
	gz.Close()
	tarPath := filepath.Join(dir, "a.tar.gz.1234")
	os.WriteFile(tarPath, tgz.Bytes(), 0o644)

	var zb bytes.Buffer
	zw := zip.NewWriter(&zb)
	for name, body := range map[string]string{"other.dll": "no", "ggml-vulkan.dll": "backend"} {
		w, _ := zw.Create(name)
		w.Write([]byte(body))
	}
	zw.Close()
	zipPath := filepath.Join(dir, "a.zip.1234")
	os.WriteFile(zipPath, zb.Bytes(), 0o644)

	for _, c := range []struct{ archive, name string }{{tarPath, "libggml-vulkan.so"}, {zipPath, "ggml-vulkan.dll"}} {
		dest := filepath.Join(dir, c.name)
		if err := extractMember(c.archive, c.name, dest); err != nil {
			t.Fatalf("%s: %v", c.archive, err)
		}
		if b, _ := os.ReadFile(dest); string(b) != "backend" {
			t.Errorf("%s: got %q", c.name, b)
		}
		if err := extractMember(c.archive, "absent", filepath.Join(dir, "x")); err == nil {
			t.Errorf("%s: a member it does not hold was found", c.archive)
		}
	}
}

func TestEveryPlatformHasAPinnedAsset(t *testing.T) {
	for _, p := range []string{"linux/amd64", "linux/arm64", "windows/amd64", "windows/arm64"} {
		a, ok := vulkanAssets[p]
		if !ok || len(a.SHA256) != 64 || a.Size == 0 || a.File == "" {
			t.Errorf("%s: %+v", p, a)
		}
	}
}
