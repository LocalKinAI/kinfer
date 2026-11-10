package server

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// imageDir makes a picture model's directory as mflux saves Z-Image.
func imageDir(t *testing.T, root, name string) string {
	t.Helper()
	dir := filepath.Join(root, name)
	files := map[string]string{
		"transformer/0.safetensors": "", "text_encoder/0.safetensors": "", "vae/0.safetensors": "",
		"tokenizer/tokenizer.json":                 "{}",
		"transformer/model.safetensors.index.json": `{"metadata":{"quantization_level":"8"},"weight_map":{"all_x_embedder.2-1.weight":"0.safetensors","cap_embedder.1.weight":"0.safetensors"}}`,
	}
	for f, body := range files {
		os.MkdirAll(filepath.Dir(filepath.Join(dir, f)), 0o755)
		if err := os.WriteFile(filepath.Join(dir, f), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// A picture request goes to the picture model's runner: the one it names if
// that paints, else this machine's default; one missing is fetched.
func TestImagesGoToTheirRunner(t *testing.T) {
	srv, f, _, _ := newFrontServer(t)
	defer srv.Close()
	img := imageDir(t, srv.store.Root(), "z-image-turbo-mflux-q8")
	srv.speechDefaults = append(srv.speechDefaults, SpeechModel{kindImage, "mflux-community/z-image-turbo-mflux-q8", "z-image-turbo-mflux-q8", "11.0 GB"})

	code, out, _ := do(t, srv, "POST", "/v1/images/generations", "application/json", `{"model":"dall-e-3","prompt":"a cat"}`)
	if code != 200 || out["model"] != img || out["path"] != "/v1/images/generations" {
		t.Fatalf("%d %v", code, out)
	}
	if _, out, _ = do(t, srv, "POST", "/v1/images/generations", "application/json", `{"model":"z-image-turbo-mflux-q8","prompt":"a cat"}`); out["model"] != img {
		t.Errorf("by its name: %v", out)
	}
	// A chat model named for a picture is not one that paints: the default.
	if _, out, _ = do(t, srv, "POST", "/v1/images/generations", "application/json", `{"model":"chat","prompt":"a cat"}`); out["model"] != img {
		t.Errorf("a chat model named: %v", out)
	}
	if f.count(f.started, img) != 1 {
		t.Errorf("runners started: %v", f.started)
	}

	os.RemoveAll(img)
	var mu sync.Mutex
	var fetched []string
	srv.pulls.start = func(p *pull) {
		mu.Lock()
		fetched = append(fetched, p.model.Repo)
		mu.Unlock()
	}
	f.live[img].stop()
	code, out, _ = do(t, srv, "POST", "/v1/images/generations", "application/json", `{"prompt":"a cat"}`)
	msg, _ := out["error"].(map[string]any)["message"].(string)
	if code != 503 || !strings.Contains(msg, "z-image-turbo-mflux-q8 is being fetched for text to image (11.0 GB)") {
		t.Errorf("missing: %d %q", code, msg)
	}
	for i := 0; i < 100; i++ { // the fetch starts in a goroutine of its own
		mu.Lock()
		n := len(fetched)
		mu.Unlock()
		if n > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	mu.Lock()
	if len(fetched) != 1 || fetched[0] != "mflux-community/z-image-turbo-mflux-q8" {
		t.Errorf("fetched %v", fetched)
	}
	mu.Unlock()
}

// What a request may ask for is checked before a model is loaded; a model
// that does not paint says so.
func TestImageRequestChecked(t *testing.T) {
	srv, _ := newTestServer(t, "chat")
	defer srv.Close()
	for body, want := range map[string]string{
		`{"model":"chat"}`:   "nothing to paint",
		`{"prompt":"a cat"}`: "no model",
		`{"model":"chat","prompt":"a cat","size":"big"}`:             "write width x height",
		`{"model":"chat","prompt":"a cat","n":9}`:                    `"n" is 1 to 4`,
		`{"model":"chat","prompt":"a cat","response_format":"jpeg"}`: "b64_json",
		`{"model":"chat","prompt":"a cat"}`:                          "does not paint",
	} {
		code, out, _ := do(t, srv, "POST", "/v1/images/generations", "application/json", body)
		msg, _ := out["error"].(map[string]any)["message"].(string)
		if code != 400 || !strings.Contains(msg, want) {
			t.Errorf("%s: %d %q, want %q", body, code, msg, want)
		}
	}
	for s, want := range map[string][2]int{"": {1024, 1024}, "auto": {1024, 1024}, "768x1024": {768, 1024}, "1536X640": {1536, 640}} {
		w, h, err := parseSize(s)
		if err != nil || w != want[0] || h != want[1] {
			t.Errorf("parseSize(%q) = %d %d %v", s, w, h, err)
		}
	}
}
