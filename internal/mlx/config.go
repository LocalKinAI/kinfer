package mlx

import (
	"math"
	"os"
	"path/filepath"

	"github.com/LocalKinAI/kinfer/internal/dl"
	"github.com/ebitengine/purego"
)

// MLX hands the GPU its work in Metal command buffers, and closes one after
// 40 or 50 operations or as many megabytes of arrays, whichever comes first
// (MLX_MAX_OPS_PER_BUFFER, MLX_MAX_MB_PER_BUFFER). With more than ten of them
// not yet finished it waits for one before it goes on. A decode step of a
// large model reads tens of gigabytes of weights, so it is hundreds of
// buffers: the step after it, started while it runs, cannot be put in before
// it has all but finished, and whatever the host then does before the next —
// reading the tokens, recording the next step's graph — the GPU sits out. On
// an M3 Ultra a step of Qwen3.8-27B took 35 ms of which the GPU worked 27.
//
// EvalStep and AsyncEvalStep evaluate with buffers large enough that a whole
// step fits in a few, so the next can be queued behind it. They are for
// passes of a few tokens only: what a buffer's operations made stays
// allocated until the buffer is done, which for a step is next to nothing
// and for a prompt of thousands of tokens is tens of gigabytes (reading
// 6,000 tokens into that model held 41 GB at its peak with these sizes
// against 3.7 GB with MLX's own).
const (
	stepOps = 400
	stepMB  = 2048
)

var (
	configUpdate func(name string, v int32)
	configGet    func(name string, def int32) int32

	bufKnown       bool // defOps and defMB read, or found unreadable
	bufOK          bool
	defOps, defMB  int32
	opsName        = "MLX_MAX_OPS_PER_BUFFER"
	mbName         = "MLX_MAX_MB_PER_BUFFER"
	stepBuffersOff = os.Getenv("KINFER_MLX_STEP_BUFFERS") == "off"
)

// bindConfig finds MLX's own settings in libmlx: C++ functions of (const
// char*, int), which mlx-c does not pass on. Without them — another MLX —
// EvalStep is Eval.
func bindConfig(dir string) {
	lib, err := dl.Open(filepath.Join(dir, "libmlx.dylib"))
	if err != nil {
		return
	}
	update, err1 := dl.Sym(lib, "_ZN3mlx4core6config6updateEPKci")
	get, err2 := dl.Sym(lib, "_ZN3mlx4core6config3getEPKci")
	if err1 != nil || err2 != nil {
		return
	}
	purego.RegisterFunc(&configUpdate, update)
	purego.RegisterFunc(&configGet, get)
}

// stepBuffers has MLX use a step's buffer sizes until the function it
// returns is called. MLX's own sizes depend on the chip and are whatever it
// has used so far, which it has by the time anything is evaluated; someone
// who set them in the environment keeps theirs.
func stepBuffers() (restore func()) {
	if !bufKnown {
		bufKnown = true
		if configUpdate != nil && !stepBuffersOff && os.Getenv(opsName) == "" && os.Getenv(mbName) == "" {
			one := Scalar(1)
			two := Add(one, one)
			two.Eval()
			defOps, defMB = configGet(opsName, -1), configGet(mbName, -1)
			if bufOK = defOps > 0 && defMB > 0; !bufOK {
				// Asking wrote our -1 down as the setting; put back the
				// sizes MLX gives most chips.
				configUpdate(opsName, 40)
				configUpdate(mbName, 40)
			}
		}
	}
	if !bufOK {
		return func() {}
	}
	configUpdate(opsName, max(defOps, stepOps))
	configUpdate(mbName, max(defMB, stepMB))
	return func() {
		configUpdate(opsName, defOps)
		configUpdate(mbName, defMB)
	}
}

// EvalStep is Eval for a decode step or a pass of a few tokens.
func EvalStep(arrays ...Array) {
	defer stepBuffers()()
	Eval(arrays...)
}

// AsyncEvalStep is AsyncEval for a decode step or a pass of a few tokens.
func AsyncEvalStep(arrays ...Array) {
	defer stepBuffers()()
	AsyncEval(arrays...)
}

// constKey names a constant: a float32's bits or an int, and its type.
type constKey struct {
	bits uint32
	dt   Dtype
	i    bool
}

var consts = map[constKey]Array{}

// Const is a 0-d array of v as dt that lives as long as the process: the
// same one each time it is asked for. Making a scalar array is an allocation
// on the GPU's side, some microseconds, and a model's step makes hundreds of
// the same few. It is for values that are fixed — a weight's scale, a
// norm's factor — and past a few hundred different ones gives an ordinary
// array, the scope's to free.
func Const(v float32, dt Dtype) Array {
	return constant(constKey{math.Float32bits(v), dt, false}, func() Array {
		a := arrayNewF32(v)
		if dt == Float32 {
			return a
		}
		r := arrayNew()
		check(opAstype(&r, a, dt, gpu))
		arrayFree(a)
		return r
	})
}

// ConstInt is Const for a 0-d int32, which a kernel takes as a scalar.
func ConstInt(v int) Array {
	return constant(constKey{uint32(int32(v)), Int32, true}, func() Array { return arrayNewInt(int32(v)) })
}

func constant(k constKey, make func() Array) Array {
	if a, ok := consts[k]; ok {
		return a
	}
	a := make()
	if len(consts) >= 512 {
		track(a)
		return a
	}
	a.Eval()
	consts[k] = a
	return a
}
