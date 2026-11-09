package store

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

func newStore(t *testing.T, files ...string) *Store {
	t.Helper()
	s, err := OpenAt(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	for i, f := range files {
		path := filepath.Join(s.Root(), f)
		if err := os.WriteFile(path, []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
		// Distinct mtimes so List's ordering is deterministic.
		mt := time.Now().Add(time.Duration(i) * time.Minute)
		if err := os.Chtimes(path, mt, mt); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

func TestList_NewestFirstAndGgufOnly(t *testing.T) {
	s := newStore(t, "old.gguf", "new.gguf")
	if err := os.WriteFile(filepath.Join(s.Root(), "notes.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	models, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 {
		t.Fatalf("List returned %d models, want 2 (non-gguf must be ignored)", len(models))
	}
	if models[0].Name != "new" {
		t.Errorf("newest first: got %q", models[0].Name)
	}
	if models[0].Size != 4 {
		t.Errorf("size = %d, want 4", models[0].Size)
	}
}

func TestList_Empty(t *testing.T) {
	models, err := newStore(t).List()
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 0 {
		t.Errorf("empty store returned %d models", len(models))
	}
}

func TestResolve(t *testing.T) {
	s := newStore(t, "qwen2.5-0.5b-instruct-q4_k_m.gguf", "llama-3.2-3b.gguf")

	cases := []struct{ in, wantFile string }{
		{"qwen2.5-0.5b-instruct-q4_k_m", "qwen2.5-0.5b-instruct-q4_k_m.gguf"},      // exact name
		{"qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen2.5-0.5b-instruct-q4_k_m.gguf"}, // filename
		{"QWEN2.5-0.5B-INSTRUCT-Q4_K_M", "qwen2.5-0.5b-instruct-q4_k_m.gguf"},      // case-insensitive
		{"qwen", "qwen2.5-0.5b-instruct-q4_k_m.gguf"},                              // unique prefix
		{"llama", "llama-3.2-3b.gguf"},
	}
	for _, c := range cases {
		got, err := s.Resolve(c.in)
		if err != nil {
			t.Errorf("Resolve(%q): %v", c.in, err)
			continue
		}
		if filepath.Base(got) != c.wantFile {
			t.Errorf("Resolve(%q) = %q, want %q", c.in, filepath.Base(got), c.wantFile)
		}
	}
}

// Loading the wrong 7B model silently is worse than an error.
func TestResolve_AmbiguousPrefixFails(t *testing.T) {
	s := newStore(t, "qwen2.5-0.5b.gguf", "qwen2.5-7b.gguf")

	_, err := s.Resolve("qwen")
	if err == nil {
		t.Fatal("ambiguous prefix should fail")
	}
	if !strings.Contains(err.Error(), "ambiguous") {
		t.Errorf("error should say what went wrong, got: %v", err)
	}
}

func TestResolve_Missing(t *testing.T) {
	s := newStore(t, "qwen.gguf")
	if _, err := s.Resolve("mistral"); err == nil {
		t.Error("Resolve on a missing model should fail")
	}
	if _, err := s.Resolve(""); err == nil {
		t.Error("Resolve(\"\") should fail")
	}
}

func TestResolve_EmptyStoreSuggestsPull(t *testing.T) {
	_, err := newStore(t).Resolve("anything")
	if err == nil || !strings.Contains(err.Error(), "pull") {
		t.Errorf("empty store should point at `pull`, got: %v", err)
	}
}

func TestRemove(t *testing.T) {
	s := newStore(t, "doomed.gguf", "keeper.gguf")

	if err := s.Remove("doomed"); err != nil {
		t.Fatal(err)
	}
	models, _ := s.List()
	if len(models) != 1 || models[0].Name != "keeper" {
		t.Errorf("after Remove, models = %v", names(models))
	}
	if err := s.Remove("doomed"); err == nil {
		t.Error("removing a gone model should fail")
	}
}

// `kinfer rm` must not become a general-purpose file deleter.
func TestRemove_RefusesOutsideStore(t *testing.T) {
	s := newStore(t)

	outside := filepath.Join(t.TempDir(), "precious.gguf")
	if err := os.WriteFile(outside, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := s.Remove(outside); err == nil {
		t.Error("Remove outside the store should be refused")
	}
	if _, err := os.Stat(outside); err != nil {
		t.Error("the outside file was deleted")
	}
}

func TestHumanSize(t *testing.T) {
	cases := map[int64]string{
		512:                    "512 B",
		1024:                   "1.0 KB",
		1024 * 1024:            "1.0 MB",
		5 * 1024 * 1024 * 1024: "5.0 GB",
	}
	for in, want := range cases {
		if got := HumanSize(in); got != want {
			t.Errorf("HumanSize(%d) = %q, want %q", in, got, want)
		}
	}
}

// Anything past roughly 50 GB arrives from Hugging Face in pieces, so a store
// that cannot recognise them cannot hold a large model at all: three files look
// like three models, each reporting a fraction of the size, and asking for the
// model by name is ambiguous between its own pieces.

func writeShards(t *testing.T, dir, base string, n int, each int) {
	t.Helper()
	for i := 1; i <= n; i++ {
		name := fmt.Sprintf("%s-%05d-of-%05d.gguf", base, i, n)
		if err := os.WriteFile(filepath.Join(dir, name), make([]byte, each), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func TestListCollapsesShardsIntoOneModel(t *testing.T) {
	dir := t.TempDir()
	writeShards(t, dir, "Qwen3.8-Flash-Next-UD-Q2_K_XL", 3, 1000)
	if err := os.WriteFile(filepath.Join(dir, "plain.gguf"), make([]byte, 7), 0o644); err != nil {
		t.Fatal(err)
	}

	s, err := OpenAt(dir)
	if err != nil {
		t.Fatal(err)
	}
	models, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 {
		t.Fatalf("List returned %d models, want 2 (one split, one plain): %+v", len(models), models)
	}

	var split *Model
	for i := range models {
		if models[i].Name == "Qwen3.8-Flash-Next-UD-Q2_K_XL" {
			split = &models[i]
		}
	}
	if split == nil {
		t.Fatalf("the split model is not listed under its base name: %+v", models)
	}
	if split.Size != 3000 {
		t.Errorf("size = %d, want the sum of all pieces (3000)", split.Size)
	}
	if !strings.HasSuffix(split.Path, "-00001-of-00003.gguf") {
		t.Errorf("path = %q, want the first piece — llama.cpp finds the rest from it", split.Path)
	}
}

func TestResolveFindsASplitModelByPrefix(t *testing.T) {
	dir := t.TempDir()
	writeShards(t, dir, "Qwen3.8-Flash-Next-UD-Q2_K_XL", 3, 10)
	s, _ := OpenAt(dir)

	got, err := s.Resolve("qwen3.8")
	if err != nil {
		t.Fatalf("Resolve by prefix failed: %v — pieces of one model must not be ambiguous with each other", err)
	}
	if !strings.HasSuffix(got, "-00001-of-00003.gguf") {
		t.Errorf("Resolve returned %q, want the first piece", got)
	}
}

func TestListIgnoresASetMissingItsFirstPiece(t *testing.T) {
	dir := t.TempDir()
	writeShards(t, dir, "Broken", 3, 10)
	if err := os.Remove(filepath.Join(dir, "Broken-00001-of-00003.gguf")); err != nil {
		t.Fatal(err)
	}
	s, _ := OpenAt(dir)
	models, _ := s.List()
	if len(models) != 0 {
		t.Errorf("a set with no first piece was listed as loadable: %+v", models)
	}
}

func TestRemoveDeletesEveryPiece(t *testing.T) {
	dir := t.TempDir()
	writeShards(t, dir, "Big", 3, 10)
	s, _ := OpenAt(dir)

	if err := s.Remove("Big"); err != nil {
		t.Fatal(err)
	}
	left, _ := os.ReadDir(dir)
	if len(left) != 0 {
		// Removing one piece would strand tens of gigabytes that nothing loads.
		t.Errorf("rm left %d pieces behind", len(left))
	}
}

// Linking a model into the store is a reasonable thing to do — it is how one
// file is shared with another runtime instead of copied. Reading the link's own
// size instead of the model's reports twenty gigabytes as a hundred bytes, and
// every size-based decision downstream is then made about nothing.
func TestListFollowsSymlinks(t *testing.T) {
	target := filepath.Join(t.TempDir(), "real-model.gguf")
	if err := os.WriteFile(target, make([]byte, 4096), 0o644); err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	if err := os.Symlink(target, filepath.Join(dir, "linked.gguf")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	s, _ := OpenAt(dir)
	models, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 {
		t.Fatalf("List returned %d models, want 1", len(models))
	}
	if models[0].Size != 4096 {
		t.Errorf("size = %d, want the target's 4096 — a link's own size is not the model's", models[0].Size)
	}
}

func TestListSkipsBrokenSymlinks(t *testing.T) {
	dir := t.TempDir()
	if err := os.Symlink(filepath.Join(dir, "gone.gguf"), filepath.Join(dir, "dangling.gguf")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	s, _ := OpenAt(dir)
	models, _ := s.List()
	if len(models) != 0 {
		t.Errorf("a link with nothing at the end of it was listed: %+v", models)
	}
}

// Ollama's store belongs to Ollama. kinfer reads models out of it so that
// someone who already has forty gigabytes on disk is not made to download them
// again — but it must never write there, and in particular `kinfer rm` must not
// be able to delete another program's data.
func TestRemoveRefusesToDeleteOllamasBlobs(t *testing.T) {
	dir := t.TempDir()
	ollama := filepath.Join(t.TempDir(), "blobs")
	if err := os.MkdirAll(ollama, 0o755); err != nil {
		t.Fatal(err)
	}
	blob := filepath.Join(ollama, "sha256-abc123")
	if err := os.WriteFile(blob, []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}

	s, _ := OpenAt(dir)
	err := s.Remove(blob)
	if err == nil {
		t.Fatal("kinfer rm deleted a file from Ollama's blob store")
	}
	if !strings.Contains(err.Error(), "ollama rm") {
		t.Errorf("error does not point at the right tool: %v", err)
	}
	if _, statErr := os.Stat(blob); statErr != nil {
		t.Error("the blob was removed despite the error")
	}
}

func TestAllPrefersKinfersOwnCopy(t *testing.T) {
	// A model present in both places is kinfer's: someone put it there on
	// purpose, and a borrowed one should not shadow it.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "shared.gguf"), make([]byte, 99), 0o644); err != nil {
		t.Fatal(err)
	}
	s, _ := OpenAt(dir)
	all, err := s.All()
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range all {
		if m.Name == "shared" && m.Source != FromKinfer {
			t.Errorf("shared model came from %q, want %q", m.Source, FromKinfer)
		}
	}
}

// OpenAt names one directory and means it. If it reached into Ollama's store as
// well, a caller asking about a path would get models from somewhere else, and
// every test here would depend on what the machine running it has installed.
func TestOpenAtDoesNotBorrow(t *testing.T) {
	s, err := OpenAt(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	all, err := s.All()
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 0 {
		t.Errorf("OpenAt on an empty directory returned %d models: %+v", len(all), all)
	}
}

// writePart leaves what a stopped pull leaves: name.part, as long as the whole
// file and holding done of it, and the record beside it.
func writePart(t *testing.T, dir, name string, size int64, done ...int64) {
	t.Helper()
	f, err := os.Create(filepath.Join(dir, name+".part"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := f.Truncate(size); err != nil {
		t.Fatal(err)
	}
	rec := fmt.Sprintf(`{"size":%d,"etag":"e","piece":%d,"done":%s}`, size, size, strings.Join(strings.Fields(fmt.Sprint(done)), ","))
	if err := os.WriteFile(filepath.Join(dir, name+".part.json"), []byte(rec), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestListLeavesOutASetMissingAPiece(t *testing.T) {
	dir := t.TempDir()
	writeShards(t, dir, "Big", 3, 10)
	if err := os.Remove(filepath.Join(dir, "Big-00002-of-00003.gguf")); err != nil {
		t.Fatal(err)
	}
	s, _ := OpenAt(dir)
	if models, _ := s.List(); len(models) != 0 {
		t.Errorf("a set missing a piece was listed as loadable: %+v", models)
	}
	unfinished, _ := s.Unfinished()
	if len(unfinished) != 1 || unfinished[0].Name != "Big" || unfinished[0].Size != 20 || unfinished[0].Total != 0 || len(unfinished[0].Files) != 2 {
		t.Errorf("Unfinished = %+v, want Big with its two pieces, 20 bytes, its whole size unknown", unfinished)
	}
}

func TestUnfinishedShowsWhatAPullLeft(t *testing.T) {
	dir := t.TempDir()
	writePart(t, dir, "Small-Q4_K_M.gguf", 1000, 300)
	writeShards(t, dir, "Big", 2, 50)
	if err := os.Remove(filepath.Join(dir, "Big-00002-of-00002.gguf")); err != nil {
		t.Fatal(err)
	}
	writePart(t, dir, "Big-00002-of-00002.gguf", 50, 10)

	s, _ := OpenAt(dir)
	if models, _ := s.List(); len(models) != 0 {
		t.Errorf("unfinished models were listed as loadable: %+v", models)
	}
	unfinished, err := s.Unfinished()
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]Unfinished{}
	for _, u := range unfinished {
		got[u.Name] = u
	}
	if u := got["Small-Q4_K_M"]; u.Size != 300 || u.Total != 1000 || len(u.Files) != 2 {
		t.Errorf("a part 300 bytes into 1000 shows as %+v", u)
	}
	// Piece 1 whole, and 10 bytes of piece 2's 50.
	if u := got["Big"]; u.Size != 60 || u.Total != 100 || len(u.Files) != 3 {
		t.Errorf("a split model with its second piece arriving shows as %+v", u)
	}
}

func TestRemoveDeletesAnUnfinishedPull(t *testing.T) {
	dir := t.TempDir()
	writePart(t, dir, "Small-Q4_K_M.gguf", 1000, 300)
	writeShards(t, dir, "Big", 3, 10)
	if err := os.Remove(filepath.Join(dir, "Big-00003-of-00003.gguf")); err != nil {
		t.Fatal(err)
	}
	s, _ := OpenAt(dir)

	for _, name := range []string{"small-q4_k_m", "Big"} {
		if err := s.Remove(name); err != nil {
			t.Fatalf("rm %s: %v", name, err)
		}
	}
	if left, _ := os.ReadDir(dir); len(left) != 0 {
		t.Errorf("rm left %d files behind", len(left))
	}
}

// A model being pulled again — a new version of it upstream — is removed
// together with what arrived of the new one.
func TestRemoveTakesAModelsUnfinishedPullWithIt(t *testing.T) {
	s := newStore(t, "Small-Q4_K_M.gguf")
	writePart(t, s.Root(), "Small-Q4_K_M.gguf", 1000, 300)
	if models, _ := s.List(); len(models) != 1 {
		t.Fatalf("the finished model is not listed: %+v", models)
	}
	if err := s.Remove("Small-Q4_K_M"); err != nil {
		t.Fatal(err)
	}
	if left, _ := os.ReadDir(s.Root()); len(left) != 0 {
		t.Errorf("rm left %d files behind", len(left))
	}
}

// Asked for by its exact name, an unfinished model is said to be unfinished —
// not run as some other model the name happens to begin.
func TestResolveSaysAModelIsUnfinished(t *testing.T) {
	s := newStore(t, "Small-Q4_K_M-imatrix.gguf")
	writePart(t, s.Root(), "Small-Q4_K_M.gguf", 1000, 300)
	_, err := s.Resolve("Small-Q4_K_M")
	if err == nil || !strings.Contains(err.Error(), "not finished") {
		t.Errorf("Resolve of an unfinished model returned %v", err)
	}
	if err := s.Remove("Small-Q4_K_M"); err != nil {
		t.Fatal(err)
	}
	if models, _ := s.List(); len(models) != 1 {
		t.Errorf("rm of an unfinished model took the model its name begins: %+v", models)
	}
}

// A vision model's projector sits beside it as NAME.mmproj.gguf, and is not a
// model of its own.
func TestProjectorBesideAModel(t *testing.T) {
	s := newStore(t, "Vis-Q4_K_M.gguf", "Vis-Q4_K_M.mmproj.gguf", "Plain.gguf", "mmproj-F16.gguf")
	models, _ := s.List()
	if len(models) != 2 {
		t.Fatalf("List = %v; a projector is not a model", names(models))
	}
	want := filepath.Join(s.Root(), "Vis-Q4_K_M.mmproj.gguf")
	for _, m := range models {
		if (m.Name == "Vis-Q4_K_M") != (m.Projector == want) {
			t.Errorf("%s has projector %q", m.Name, m.Projector)
		}
	}
	if got := s.ProjectorFor(filepath.Join(s.Root(), "Vis-Q4_K_M.gguf")); got != want {
		t.Errorf("ProjectorFor = %q, want %q", got, want)
	}
	if got := s.ProjectorFor(filepath.Join(s.Root(), "Plain.gguf")); got != "" {
		t.Errorf("a text model's projector is %q", got)
	}

	writeShards(t, s.Root(), "Big", 2, 10)
	if err := os.WriteFile(filepath.Join(s.Root(), "Big.mmproj.gguf"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := s.ProjectorFor(filepath.Join(s.Root(), "Big-00001-of-00002.gguf")); got != filepath.Join(s.Root(), "Big.mmproj.gguf") {
		t.Errorf("a split model's projector is %q", got)
	}
}

// Ollama keeps a vision model's projector as a layer of its own; a model linked
// into kinfer's store from there finds it too.
func TestOllamaProjector(t *testing.T) {
	root := t.TempDir()
	blobs := filepath.Join(root, "blobs")
	manifests := filepath.Join(root, "manifests")
	path := filepath.Join(manifests, "registry.ollama.ai", "library", "vis", "latest")
	for _, dir := range []string{blobs, filepath.Dir(path)} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	// Ollama's store is told from an empty one by weights over a megabyte;
	// sparse, they take no disk.
	for name, size := range map[string]int64{"sha256-aaa": 2 << 20, "sha256-bbb": 4} {
		f, err := os.Create(filepath.Join(blobs, name))
		if err == nil {
			err = f.Truncate(size)
			f.Close()
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	manifest := `{"layers":[
	  {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:aaa","size":2097152},
	  {"mediaType":"application/vnd.ollama.image.projector","digest":"sha256:bbb","size":4}]}`
	if err := os.WriteFile(path, []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	m, ok := readOllamaManifest(root, manifests, path)
	if !ok || m.Name != "vis:latest" || m.Projector != filepath.Join(blobs, "sha256-bbb") {
		t.Fatalf("readOllamaManifest = %+v, %v", m, ok)
	}

	t.Setenv("OLLAMA_MODELS", root)
	s, _ := OpenAt(t.TempDir())
	s.borrow = true
	link := filepath.Join(s.Root(), "vis.gguf")
	if err := os.Symlink(filepath.Join(blobs, "sha256-aaa"), link); err != nil {
		t.Fatal(err)
	}
	if got := s.ProjectorFor(link); got != m.Projector {
		t.Errorf("a link to Ollama's blob found projector %q, want %q", got, m.Projector)
	}
}

// Removing a vision model removes the projector kinfer keeps beside it.
func TestRemoveTakesTheProjector(t *testing.T) {
	s := newStore(t, "Vis-Q4_K_M.gguf", "Vis-Q4_K_M.mmproj.gguf", "Other.gguf")
	if err := s.Remove("Vis-Q4_K_M"); err != nil {
		t.Fatal(err)
	}
	left, _ := os.ReadDir(s.Root())
	if len(left) != 1 || left[0].Name() != "Other.gguf" {
		var got []string
		for _, e := range left {
			got = append(got, e.Name())
		}
		t.Errorf("rm left %v", got)
	}
}

// fakeOllama makes root an Ollama store holding library/NAME:latest, whose
// weights are a sparse 2 MiB blob, and returns that blob's path.
func fakeOllama(t *testing.T, root, name string) string {
	t.Helper()
	blob := filepath.Join(root, "blobs", "sha256-"+name)
	manifest := filepath.Join(root, "manifests", "registry.ollama.ai", "library", name, "latest")
	for _, dir := range []string{filepath.Dir(blob), filepath.Dir(manifest)} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	f, err := os.Create(blob)
	if err == nil {
		err = f.Truncate(2 << 20)
		f.Close()
	}
	if err != nil {
		t.Fatal(err)
	}
	m := `{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:` + name + `","size":2097152}]}`
	if err := os.WriteFile(manifest, []byte(m), 0o644); err != nil {
		t.Fatal(err)
	}
	return blob
}

// An MLX model of Ollama's is a manifest naming a blob for each tensor and
// for each of its JSON files. kinfer lists it as a directory of links to
// those blobs, with the chat template written out, made again only when the
// manifest changes, and never deletes it.
func TestOllamaMLXModel(t *testing.T) {
	root, home := t.TempDir(), t.TempDir()
	t.Setenv("OLLAMA_MODELS", root)
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("XDG_CACHE_HOME", filepath.Join(home, ".cache"))
	blob := func(digest, content string, size int64) string {
		p := filepath.Join(root, "blobs", "sha256-"+digest)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		f, err := os.Create(p)
		if err == nil {
			_, err = f.WriteString(content)
		}
		if err == nil && size > 0 {
			err = f.Truncate(size)
		}
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	weight := blob("w", "", 2<<20)
	blob("c", `{"model_type":"qwen3_5"}`, 0)
	blob("t", `{}`, 0)
	blob("tc", `{"chat_template":"{{ messages }}"}`, 0)
	manifest := filepath.Join(root, "manifests", "registry.ollama.ai", "library", "qwen3.8", "27b-mlx")
	if err := os.MkdirAll(filepath.Dir(manifest), 0o755); err != nil {
		t.Fatal(err)
	}
	layers := `{"mediaType":"application/vnd.ollama.image.tensor","digest":"sha256:w","size":2097152,"name":"model.layers.0.mlp.up_proj.weight"},` +
		`{"mediaType":"application/vnd.ollama.image.json","digest":"sha256:c","size":24,"name":"config.json"},` +
		`{"mediaType":"application/vnd.ollama.image.json","digest":"sha256:t","size":2,"name":"tokenizer.json"},` +
		`{"mediaType":"application/vnd.ollama.image.json","digest":"sha256:tc","size":33,"name":"tokenizer_config.json"}`
	write := func(layers string) {
		if err := os.WriteFile(manifest, []byte(`{"layers":[`+layers+`]}`), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	find := func() Model {
		for _, m := range OllamaModels() {
			if m.Name == "qwen3.8:27b-mlx" {
				return m
			}
		}
		t.Fatalf("Ollama's MLX model is not listed: %+v", OllamaModels())
		return Model{}
	}
	write(layers)

	m := find()
	if m.Format != "safetensors" || m.Source != FromOllama || m.Size != 2<<20 || !strings.HasPrefix(m.Path, OllamaMLXRoot()) {
		t.Fatalf("listed as %+v", m)
	}
	if got, err := os.Readlink(filepath.Join(m.Path, "model.layers.0.mlp.up_proj.weight.safetensors")); err != nil || got != weight {
		t.Errorf("the tensor's link points at %q, %v; want %s", got, err, weight)
	}
	if b, err := os.ReadFile(filepath.Join(m.Path, "config.json")); err != nil || !strings.Contains(string(b), "qwen3_5") {
		t.Errorf("config.json reads %q, %v", b, err)
	}
	if b, err := os.ReadFile(filepath.Join(m.Path, "chat_template.jinja")); err != nil || string(b) != "{{ messages }}" {
		t.Errorf("chat_template.jinja reads %q, %v", b, err)
	}

	// Listed again, the directory is the one already made.
	mark := filepath.Join(m.Path, "mark")
	if err := os.WriteFile(mark, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if find(); !exists(mark) {
		t.Error("the directory was made again for the same manifest")
	}
	// A new manifest — the model pulled again — makes it again.
	write(layers + `,{"mediaType":"application/vnd.ollama.image.json","digest":"sha256:t","size":2,"name":"vocab.json"}`)
	if m2 := find(); exists(mark) || !exists(filepath.Join(m2.Path, "vocab.json")) {
		t.Error("a changed manifest did not make the directory again")
	}

	// A directory made while tokenizer_config.json was still being copied in
	// has no template; once the file is whole, the next listing makes it again.
	tc := filepath.Join(root, "blobs", "sha256-tc")
	if err := os.WriteFile(tc, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	write(layers + `,{"mediaType":"application/vnd.ollama.image.json","digest":"sha256:t","size":2,"name":"merges.txt"}`)
	if m3 := find(); exists(filepath.Join(m3.Path, "chat_template.jinja")) {
		t.Fatal("a template was written from an empty tokenizer_config.json")
	}
	blob("tc", `{"chat_template":"{{ messages }}"}`, 0)
	if m3 := find(); !exists(filepath.Join(m3.Path, "chat_template.jinja")) {
		t.Error("the directory kept no template after tokenizer_config.json came whole")
	}

	s := newStore(t)
	s.borrow = true
	if err := s.Remove("qwen3.8:27b-mlx"); err == nil || !strings.Contains(err.Error(), "ollama rm") {
		t.Errorf("rm of Ollama's MLX model: %v, want a pointer to ollama rm", err)
	}
	if !exists(weight) || !exists(filepath.Join(m.Path, "config.json")) {
		t.Error("rm took something of Ollama's")
	}
}

func exists(path string) bool { _, err := os.Stat(path); return err == nil }

// An Ollama client's bare name is its :latest tag, before any model the name
// begins — as on the box, where "qwen3.8" also begins two of kinfer's own.
func TestResolveTakesABareNameAsLatest(t *testing.T) {
	root := t.TempDir()
	blob := fakeOllama(t, root, "qwen3.8")
	t.Setenv("OLLAMA_MODELS", root)
	s := newStore(t, "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf", "qwen3.8-27b-q4_k_m.gguf")
	s.borrow = true

	if got, err := s.Resolve("qwen3.8"); err != nil || got != blob {
		t.Errorf("Resolve(qwen3.8) = %q, %v; want Ollama's qwen3.8:latest", got, err)
	}
	if got, err := s.Resolve("qwen3.8-27b"); err != nil || filepath.Base(got) != "qwen3.8-27b-q4_k_m.gguf" {
		t.Errorf("Resolve(qwen3.8-27b) = %q, %v; a prefix still finds the one it begins", got, err)
	}
}

// An MLX directory and a GGUF of the same weights can be named but for case.
// Each name is its own model, whichever is newer, and rm takes the one named;
// a name that is neither, but for case, is ambiguous.
func TestNamesThatDifferOnlyInCase(t *testing.T) {
	s := newStore(t, "Qwen3.5-0.8B-BF16.gguf")
	dir := writeDir(t, s, "Qwen3.5-0.8B-bf16", "config.json", "tokenizer.json", "model.safetensors")
	later := time.Now().Add(time.Hour)
	if err := os.Chtimes(filepath.Join(dir, "model.safetensors"), later, later); err != nil {
		t.Fatal(err)
	}

	for name, want := range map[string]string{"Qwen3.5-0.8B-BF16": "gguf", "Qwen3.5-0.8B-bf16": "safetensors"} {
		if m, err := s.Lookup(name); err != nil || m.Format != want {
			t.Errorf("Lookup(%s) = %s %v, want the %s", name, m.Format, err, want)
		}
	}
	if p, err := s.Resolve("Qwen3.5-0.8B-BF16"); err != nil || filepath.Base(p) != "Qwen3.5-0.8B-BF16.gguf" {
		t.Errorf("Resolve(Qwen3.5-0.8B-BF16) = %q, %v; want the GGUF", p, err)
	}
	if p, err := s.Resolve("Qwen3.5-0.8B-bf16"); err != nil || p != dir {
		t.Errorf("Resolve(Qwen3.5-0.8B-bf16) = %q, %v; want the directory", p, err)
	}
	for _, name := range []string{"qwen3.5-0.8b-bf16", "QWEN3.5-0.8B-BF16"} {
		if _, err := s.Resolve(name); err == nil || !strings.Contains(err.Error(), "ambiguous") {
			t.Errorf("Resolve(%s): %v, want ambiguous", name, err)
		}
		if _, err := s.Lookup(name); err == nil || !strings.Contains(err.Error(), "ambiguous") {
			t.Errorf("Lookup(%s): %v, want ambiguous", name, err)
		}
	}

	if err := s.Remove("Qwen3.5-0.8B-BF16"); err != nil {
		t.Fatal(err)
	}
	if models, _ := s.List(); len(models) != 1 || models[0].Format != "safetensors" {
		t.Errorf("rm Qwen3.5-0.8B-BF16 left %+v, want only the directory", models)
	}
}

// Pulled again under the other case, a model half arrived is that model:
// asked for, it is unfinished, and rm takes what arrived, not the finished
// one beside it.
func TestAnUnfinishedPullNamedButForCase(t *testing.T) {
	s := newStore(t, "Qwen3.5-0.8B-BF16.gguf")
	writeDir(t, s, "Qwen3.5-0.8B-bf16", "config.json", "model.safetensors.part")

	if _, err := s.Resolve("Qwen3.5-0.8B-bf16"); err == nil || !strings.Contains(err.Error(), "not finished") {
		t.Errorf("Resolve of the half-pulled directory: %v, want a not-finished error", err)
	}
	if p, err := s.Resolve("Qwen3.5-0.8B-BF16"); err != nil || filepath.Base(p) != "Qwen3.5-0.8B-BF16.gguf" {
		t.Errorf("Resolve(Qwen3.5-0.8B-BF16) = %q, %v; want the GGUF", p, err)
	}
	if err := s.Remove("Qwen3.5-0.8B-bf16"); err != nil {
		t.Fatal(err)
	}
	if models, _ := s.List(); len(models) != 1 || models[0].Name != "Qwen3.5-0.8B-BF16" {
		t.Errorf("rm of the unfinished directory took the GGUF: %+v", models)
	}
	if un, _ := s.Unfinished(); len(un) != 0 {
		t.Errorf("rm left the unfinished pull: %+v", un)
	}
}

// writeDir makes dir in s with the given files, each holding its own name.
func writeDir(t *testing.T, s *Store, dir string, files ...string) string {
	t.Helper()
	path := filepath.Join(s.Root(), dir)
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		if err := os.WriteFile(filepath.Join(path, f), []byte(f), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return path
}

// A directory of safetensors is a model once every file of it is here, and
// until then an unfinished pull; a directory of anything else is neither.
func TestModelDirectories(t *testing.T) {
	s := newStore(t, "small-Q4_K_M.gguf")
	index := `{"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}`
	whole := writeDir(t, s, "Qwen3.8-27B-4bit", "config.json", "tokenizer.json", "chat_template.jinja",
		"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
	if err := os.WriteFile(filepath.Join(whole, "model.safetensors.index.json"), []byte(index), 0o644); err != nil {
		t.Fatal(err)
	}
	// A pull stopped after the first piece, with the second arriving.
	half := writeDir(t, s, "Qwen3.5-0.8B", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors.part")
	if err := os.WriteFile(filepath.Join(half, "model.safetensors.index.json"), []byte(index), 0o644); err != nil {
		t.Fatal(err)
	}
	writeDir(t, s, "notes", "todo.txt")

	models, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, m := range models {
		got = append(got, m.Name+" "+m.Format)
	}
	if want := "Qwen3.8-27B-4bit safetensors, small-Q4_K_M gguf"; strings.Join(sortedCopy(got), ", ") != want {
		t.Errorf("List = %v, want %s", got, want)
	}
	for _, m := range models {
		if m.Name == "Qwen3.8-27B-4bit" {
			// Every file but the index: config, tokenizer, template and two pieces.
			want := int64(len("config.json") + len("tokenizer.json") + len("chat_template.jinja") +
				2*len("model-00001-of-00002.safetensors") + len(index))
			if m.Size != want || m.Path != whole {
				t.Errorf("directory model: size %d at %s, want %d at %s", m.Size, m.Path, want, whole)
			}
		}
	}

	un, err := s.Unfinished()
	if err != nil {
		t.Fatal(err)
	}
	if len(un) != 1 || un[0].Name != "Qwen3.5-0.8B" {
		t.Fatalf("Unfinished = %+v, want only Qwen3.5-0.8B", un)
	}
	if _, err := s.Resolve("Qwen3.5-0.8B"); err == nil || !strings.Contains(err.Error(), "not finished") {
		t.Errorf("Resolve of the half-pulled directory: %v, want a not-finished error", err)
	}
	if p, err := s.Resolve("qwen3.8-27b"); err != nil || p != whole {
		t.Errorf("Resolve(qwen3.8-27b) = %q, %v; want %s", p, err, whole)
	}

	for _, name := range []string{"Qwen3.8-27B-4bit", "Qwen3.5-0.8B"} {
		if err := s.Remove(name); err != nil {
			t.Fatalf("Remove(%s): %v", name, err)
		}
	}
	for _, dir := range []string{whole, half} {
		if _, err := os.Stat(dir); !os.IsNotExist(err) {
			t.Errorf("%s is still there after Remove", dir)
		}
	}
	if _, err := os.Stat(filepath.Join(s.Root(), "notes", "todo.txt")); err != nil {
		t.Errorf("a directory that is no model was touched: %v", err)
	}
}

func sortedCopy(s []string) []string {
	out := append([]string(nil), s...)
	sort.Strings(out)
	return out
}
