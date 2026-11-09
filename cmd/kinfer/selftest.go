package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/hardware"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
	"github.com/LocalKinAI/kinfer/internal/store"
)

// kinfer selftest runs one small model on the GPU and on the CPU and says
// whether the answers make sense, in a form to paste into an issue. It exists
// because a broken forward pass does not look broken: it goes on producing
// tokens at full speed, and only reading them shows they are wrong.

func init() { engine.GPUProbeArgs = []string{"gpu-probe"} }

// cmdGPUProbe is the child that tries a GPU backend, so that a driver which
// crashes takes only this process with it. It prints one line of JSON.
func cmdGPUProbe(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: kinfer gpu-probe <backend>")
	}
	dir, err := nativelib.Prepare()
	if err != nil {
		return err
	}
	if err := llama.Bind(dir); err != nil {
		return err
	}
	if err := llama.LoadBackend(args[0]); err != nil {
		return err
	}
	p := engine.GPUProbe{OK: true}
	for _, d := range llama.Devices() {
		p.Devices = append(p.Devices, engine.ProbeDevice{Name: d.Name, Type: d.Type.String(), Total: d.Total, Free: d.Free})
	}
	return json.NewEncoder(os.Stdout).Encode(p)
}

var selftestQuestions = []struct{ prompt, expect string }{
	{"What is the capital of France? One short sentence.", "paris"},
	{"Count from 1 to 5, separated by spaces.", "3"},
}

type selftestReply struct {
	Text string `json:"text"`
	Good bool   `json:"good"`
}

type selftestResult struct {
	Replies  []selftestReply `json:"replies"`
	TokPerS  float64         `json:"tok_per_s"`
	OnGPU    bool            `json:"on_gpu"`
	Device   string          `json:"device,omitempty"`
	LoadSecs float64         `json:"load_secs"`
}

// cmdSelftestRun is one run, in a child process so that a crash is a finding
// rather than the end of the test. It prints one line of JSON.
func cmdSelftestRun(args []string) error {
	fs := flag.NewFlagSet("selftest-run", flag.ExitOnError)
	ngl := fs.Int("ngl", 99, "layers on the GPU")
	if err := fs.Parse(args); err != nil {
		return err
	}
	st, err := store.Open()
	if err != nil {
		return err
	}
	path, err := st.Resolve(fs.Arg(0))
	if err != nil {
		return err
	}

	start := time.Now()
	eng, err := engine.Open(path, engine.Options{GPULayers: *ngl, ContextSize: 4096, Slots: 1})
	if err != nil {
		return err
	}
	defer eng.Close()
	res := selftestResult{LoadSecs: time.Since(start).Seconds(), OnGPU: eng.OnGPU()}
	if d, ok := llama.Accelerator(); ok && res.OnGPU {
		res.Device = d.Name
	}

	var tokens int
	var took time.Duration
	for _, q := range selftestQuestions {
		p := engine.DefaultGenParams()
		p.Temperature, p.MaxTokens, p.Seed, p.NoThink = 0, 48, 1, true
		text, stats, err := eng.ChatFull(context.Background(), []chat.Message{{Role: "user", Content: q.prompt}}, p, nil)
		if err != nil {
			return err
		}
		res.Replies = append(res.Replies, selftestReply{Text: text, Good: repliedSensibly(text, q.expect)})
		tokens += stats.EvalTokens
		took += stats.EvalDuration
	}
	if took > 0 {
		res.TokPerS = float64(tokens) / took.Seconds()
	}
	return json.NewEncoder(os.Stdout).Encode(res)
}

// repliedSensibly is a reply with what the question expects in it and no short
// run of text said over and over.
func repliedSensibly(text, expect string) bool {
	return strings.Contains(strings.ToLower(text), expect) && !repeatsItself(text)
}

func repeatsItself(text string) bool {
	const times = 5
	for size := 2; size <= 16; size++ {
		for i := 0; i+size*times <= len(text); i++ {
			chunk := text[i : i+size]
			if strings.TrimSpace(chunk) == "" {
				continue
			}
			if strings.HasPrefix(text[i:], strings.Repeat(chunk, times)) {
				return true
			}
		}
	}
	return false
}

// runSelftest runs one child and reports what it did.
func runSelftest(ngl int, model string) (selftestResult, error) {
	exe, err := os.Executable()
	if err != nil {
		return selftestResult{}, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	var out bytes.Buffer
	cmd := exec.CommandContext(ctx, exe, "selftest-run", "-ngl", fmt.Sprint(ngl), model)
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return selftestResult{}, err
	}
	var r selftestResult
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &r); err != nil {
		return selftestResult{}, err
	}
	return r, nil
}

func smallestModel(st *store.Store) (string, error) {
	models, err := st.List()
	if err != nil {
		return "", err
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Size < models[j].Size })
	for _, m := range models {
		if m.Format == "gguf" && m.Size > 100<<20 {
			return m.Name, nil
		}
	}
	return "", fmt.Errorf("no model to test with; try: kinfer pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m (470 MB)")
}

func showRun(label string, r selftestResult, err error) bool {
	if err != nil {
		fmt.Printf("%-5s FAILED: %v\n", label, err)
		return false
	}
	good := true
	for _, x := range r.Replies {
		good = good && x.Good
	}
	where := "CPU"
	if r.OnGPU {
		where = "GPU " + r.Device
	}
	mark := "ok"
	if !good {
		mark = "WRONG"
	}
	fmt.Printf("%-5s %-5s %.1f tok/s, loaded in %.1f s, on %s\n", label, mark, r.TokPerS, r.LoadSecs, where)
	for _, x := range r.Replies {
		text := strings.Join(strings.Fields(x.Text), " ")
		if len(text) > 100 {
			text = text[:100] + "…"
		}
		fmt.Printf("      %q\n", text)
	}
	return good
}

func cmdSelftest(args []string) error {
	st, err := store.Open()
	if err != nil {
		return err
	}
	var model string
	if len(args) > 0 {
		model = args[0]
	} else if model, err = smallestModel(st); err != nil {
		return err
	}

	info := hardware.Detect()
	if err := cmdVersion(); err != nil {
		return err
	}
	fmt.Printf("%s/%s, %d processors, %.0f GB of memory\n", runtime.GOOS, runtime.GOARCH, info.Cores, float64(info.RAM)/(1<<30))
	fmt.Printf("model: %s\n", model)

	canGPU := runtime.GOOS == "darwin"
	if runtime.GOOS != "darwin" {
		switch r := engine.CheckGPU(); {
		case strings.EqualFold(os.Getenv("KINFER_GPU"), "off"):
			fmt.Println("gpu: turned off with KINFER_GPU=off")
		case !r.Loader:
			fmt.Println("gpu: no Vulkan driver found, and no backend chosen with kinfer gpu")
		case r.Problem != "":
			fmt.Printf("gpu: could not use the %s backend: %s\n", r.Label, r.Problem)
		case !r.Probe.OK:
			fmt.Printf("gpu: the %s backend crashed or failed to start: %s\n", r.Label, r.Probe.Error)
		case len(r.Probe.GPUs()) == 0:
			fmt.Printf("gpu: %s found no usable GPU\n", r.Label)
		default:
			canGPU = true
			for _, d := range r.Probe.GPUs() {
				fmt.Printf("gpu: %s, %s, %.1f GB total, %.1f GB free (%s, experimental)\n", d.Name, d.Type, float64(d.Total)/(1<<30), float64(d.Free)/(1<<30), r.Label)
			}
		}
	}

	fmt.Println()
	cpu, cpuErr := runSelftest(0, model)
	cpuGood := showRun("CPU", cpu, cpuErr)
	if !canGPU {
		fmt.Println()
		if cpuGood {
			fmt.Println("result: the CPU path works; no GPU to test")
			return nil
		}
		return fmt.Errorf("the CPU path gave wrong answers; please report this output")
	}

	gpu, gpuErr := runSelftest(99, model)
	gpuGood := showRun("GPU", gpu, gpuErr)

	fmt.Println()
	switch {
	case gpuErr != nil && cpuGood:
		fmt.Println("result: the GPU run failed; kinfer can run on the CPU. Please report this output, and use KINFER_GPU=off for now")
	case gpuErr == nil && !gpu.OnGPU && cpuGood:
		fmt.Println("result: the model did not fit the GPU or failed on it, so it ran on the CPU")
	case !gpuGood && cpuGood:
		fmt.Println("result: the GPU gives WRONG answers and the CPU does not. Please report this output, and use KINFER_GPU=off for now")
	case gpuGood && cpuGood:
		fmt.Println("result: both work")
	default:
		return fmt.Errorf("the answers were wrong on the CPU too; please report this output")
	}
	fmt.Println("reports: https://github.com/LocalKinAI/kinfer/issues")
	return nil
}
