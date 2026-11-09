package mlx

import (
	"fmt"

	"github.com/LocalKinAI/kinfer/internal/dl"
	"github.com/ebitengine/purego"
)

// A Kernel is a Metal kernel written as source and compiled by MLX the first
// time it runs, as mx.fast.metal_kernel. MLX writes the signature from the
// input and output names, so the source is only the body; inputs keep their
// own dtypes, and a 0-d input arrives as a scalar.
type Kernel struct {
	k      uintptr // mlx_fast_metal_kernel
	inputs []string
}

type (
	strVec uintptr // mlx_vector_string
	kcfg   uintptr // mlx_fast_metal_kernel_config
)

var (
	strVecNew    func() strVec
	strVecAppend func(strVec, string) int32
	strVecFree   func(strVec) int32

	kernelNew   func(name string, in, out strVec, source, header string, rowContiguous, atomicOutputs bool) uintptr
	kernelFree  func(uintptr)
	kernelApply func(outputs *vector, k uintptr, inputs vector, cfg kcfg, s Stream) int32

	kcfgNew       func() kcfg
	kcfgFree      func(kcfg)
	kcfgAddOutput func(c kcfg, shape *int32, n uintptr, dt Dtype) int32
	kcfgGrid      func(c kcfg, x, y, z int32) int32
	kcfgGroup     func(c kcfg, x, y, z int32) int32
	kcfgTplDtype  func(c kcfg, name string, dt Dtype) int32
	kcfgTplInt    func(c kcfg, name string, v int32) int32
)

func bindKernels(lib uintptr) error {
	for _, b := range []struct {
		fn   any
		name string
	}{
		{&strVecNew, "mlx_vector_string_new"}, {&strVecAppend, "mlx_vector_string_append_value"},
		{&strVecFree, "mlx_vector_string_free"},
		{&kernelNew, "mlx_fast_metal_kernel_new"}, {&kernelFree, "mlx_fast_metal_kernel_free"},
		{&kernelApply, "mlx_fast_metal_kernel_apply"},
		{&kcfgNew, "mlx_fast_metal_kernel_config_new"}, {&kcfgFree, "mlx_fast_metal_kernel_config_free"},
		{&kcfgAddOutput, "mlx_fast_metal_kernel_config_add_output_arg"},
		{&kcfgGrid, "mlx_fast_metal_kernel_config_set_grid"},
		{&kcfgGroup, "mlx_fast_metal_kernel_config_set_thread_group"},
		{&kcfgTplDtype, "mlx_fast_metal_kernel_config_add_template_arg_dtype"},
		{&kcfgTplInt, "mlx_fast_metal_kernel_config_add_template_arg_int"},
	} {
		if _, err := dl.Sym(lib, b.name); err != nil {
			return fmt.Errorf("mlx-c has no %s: %w", b.name, err)
		}
		purego.RegisterLibFunc(b.fn, lib, b.name)
	}
	return nil
}

// NewKernel defines a kernel. It lives for the life of the process, like the
// pipeline MLX caches for it.
func NewKernel(name string, inputs, outputs []string, source string) *Kernel {
	in, out := strVecNew(), strVecNew()
	defer strVecFree(in)
	defer strVecFree(out)
	for _, s := range inputs {
		strVecAppend(in, s)
	}
	for _, s := range outputs {
		strVecAppend(out, s)
	}
	k := kernelNew(name, in, out, source, "", true, false)
	if k == 0 {
		check(1)
	}
	return &Kernel{k: k, inputs: inputs}
}

// KernelOutput is the shape and dtype of one output.
type KernelOutput struct {
	Shape []int
	Dtype Dtype
}

// KernelTemplate is a template argument: a Dtype or an int.
type KernelTemplate struct {
	Name  string
	Dtype Dtype
	Int   int
	IsInt bool
}

// Run launches the kernel over grid with threadgroup size group.
func (k *Kernel) Run(inputs []Array, outputs []KernelOutput, tpl []KernelTemplate, grid, group [3]int) []Array {
	if len(inputs) != len(k.inputs) {
		panic(&Error{fmt.Sprintf("kernel wants %d inputs, got %d", len(k.inputs), len(inputs))})
	}
	cfg := kcfgNew()
	defer kcfgFree(cfg)
	for _, o := range outputs {
		p, n := ints(o.Shape)
		check(kcfgAddOutput(cfg, p, n, o.Dtype))
	}
	for _, t := range tpl {
		if t.IsInt {
			check(kcfgTplInt(cfg, t.Name, int32(t.Int)))
		} else {
			check(kcfgTplDtype(cfg, t.Name, t.Dtype))
		}
	}
	check(kcfgGrid(cfg, int32(grid[0]), int32(grid[1]), int32(grid[2])))
	check(kcfgGroup(cfg, int32(group[0]), int32(group[1]), int32(group[2])))
	in := toVector(inputs)
	defer vecFree(in)
	res := vecNew()
	defer vecFree(res)
	check(kernelApply(&res, k.k, in, cfg, gpu))
	return fromVector(res)
}

// Int makes a 0-d int32 array, which a kernel receives as a scalar.
func Int(v int) Array { a := arrayNewInt(int32(v)); track(a); return a }
