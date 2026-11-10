// Package mlx binds Apple's MLX through its C API, mlx-c, the same way
// internal/llama binds llama.cpp: purego resolves the symbols at runtime, so
// there is still no CGO and cross-compilation keeps working.
//
// MLX arrays are lazy. Every op here only records a node in a graph; nothing
// runs on the GPU until Eval (or reading the data) asks for it. A model's
// forward pass is therefore ordinary Go code that builds one graph per step,
// and the step costs one Eval.
//
// # Threads
//
// An ordinary MLX stream registers its GPU command encoder on the thread that
// creates it, and evaluating it from any other thread fails with "There is no
// Stream(gpu, 0) in current thread". Go moves goroutines between threads
// freely, so Open makes the one stream every op uses with
// mlx_stream_new_thread_unsafe instead, which MLX registers globally. MLX then
// leaves synchronization to the caller: only one goroutine may use this
// package at a time — the engine's step loop, or a test.
//
// # Errors
//
// mlx-c's default error handler prints the message and calls exit(-1), which
// would take the whole server down on a shape mistake. Open installs a handler
// that keeps the message instead; a failed op then panics with *Error, and the
// step that ran it recovers. Ops panic rather than return errors because a
// forward pass is hundreds of them in a row, and a model written with an error
// check after each would be unreadable.
//
// # Memory
//
// Every Array holds a reference to a C++ shared_ptr and must be freed. Arrays
// made while a Scope is open belong to it and are freed when it closes, except
// those passed to Keep; weights and caches are made outside any scope, or
// kept, and freed explicitly.
package mlx

import (
	"fmt"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"

	"github.com/LocalKinAI/kinfer/internal/dl"
	"github.com/ebitengine/purego"
)

// Array is an mlx_array: a struct holding one pointer, which arm64 passes and
// returns in a register exactly as it does a uintptr.
type Array uintptr

// Stream is an mlx_stream, laid out the same way.
type Stream uintptr

// Dtype mirrors enum mlx_dtype.
type Dtype int32

const (
	Bool Dtype = iota
	Uint8
	Uint16
	Uint32
	Uint64
	Int8
	Int16
	Int32
	Int64
	Float16
	Float32
	Float64
	Bfloat16
	Complex64
)

var dtypeNames = [...]string{"bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32",
	"int64", "float16", "float32", "float64", "bfloat16", "complex64"}

func (d Dtype) String() string {
	if d >= 0 && int(d) < len(dtypeNames) {
		return dtypeNames[d]
	}
	return fmt.Sprintf("dtype(%d)", int32(d))
}

// Error is what a failed op panics with: MLX's own message.
type Error struct{ Msg string }

func (e *Error) Error() string { return "mlx: " + e.Msg }

var (
	openOnce sync.Once
	openErr  error

	// lastErr is written by the error handler and read right after the call
	// that triggered it, both on the one MLX thread.
	lastErr string

	// gpu runs every op; cpu only reads files, since MLX's Load primitive has
	// no GPU implementation.
	gpu, cpu Stream
)

var (
	setErrorHandler func(handler, data, dtor uintptr)
	deviceNewType   func(kind, index int32) uintptr
	streamNewUnsafe func(dev uintptr) Stream
	streamFree      func(Stream) int32
	synchronize     func(Stream) int32

	arrayNew      func() Array
	arrayNewData  func(data unsafe.Pointer, shape *int32, dim int32, dtype Dtype) Array
	arrayFree     func(Array) int32
	arrayEval     func(Array) int32
	arrayNdim     func(Array) uintptr
	arrayShape    func(Array) *int32
	arrayDtype    func(Array) Dtype
	arraySize     func(Array) uintptr
	arrayNbytes   func(Array) uintptr
	arrayDataF32  func(Array) *float32
	arrayItemF32  func(res *float32, a Array) int32
	arrayItemI32  func(res *int32, a Array) int32
	arrayDataI32  func(Array) *int32
	arrayNewF32   func(float32) Array
	arrayNewInt   func(int32) Array
	arraySetArray func(dst *Array, src Array) int32

	opAdd    func(res *Array, a, b Array, s Stream) int32
	opMatmul func(res *Array, a, b Array, s Stream) int32
	opAstype func(res *Array, a Array, dt Dtype, s Stream) int32
)

// LibName is the file Open loads from the directory it is given; libmlx.dylib
// and mlx.metallib must sit beside it.
const LibName = "libmlxc.dylib"

// Open loads mlx-c from dir and makes the default GPU stream. It runs once;
// later calls return the first result.
func Open(dir string) error {
	openOnce.Do(func() { openErr = open(dir) })
	return openErr
}

func open(dir string) error {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		return fmt.Errorf("mlx runs on Apple silicon only, not %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	path := filepath.Join(dir, LibName)
	lib, err := dl.Open(path)
	if err != nil {
		return fmt.Errorf("open %s: %w", path, err)
	}
	for _, b := range []struct {
		fn   any
		name string
	}{
		{&setErrorHandler, "mlx_set_error_handler"},
		{&deviceNewType, "mlx_device_new_type"},
		{&streamNewUnsafe, "mlx_stream_new_thread_unsafe"},
		{&streamFree, "mlx_stream_free"},
		{&synchronize, "mlx_synchronize"},
		{&arrayNew, "mlx_array_new"},
		{&arrayNewData, "mlx_array_new_data"},
		{&arrayFree, "mlx_array_free"},
		{&arrayEval, "mlx_array_eval"},
		{&arrayNdim, "mlx_array_ndim"},
		{&arrayShape, "mlx_array_shape"},
		{&arrayDtype, "mlx_array_dtype"},
		{&arraySize, "mlx_array_size"},
		{&arrayNbytes, "mlx_array_nbytes"},
		{&arrayDataF32, "mlx_array_data_float32"},
		{&arrayDataI32, "mlx_array_data_int32"},
		{&arrayItemF32, "mlx_array_item_float32"},
		{&arrayItemI32, "mlx_array_item_int32"},
		{&arrayNewF32, "mlx_array_new_float32"},
		{&arrayNewInt, "mlx_array_new_int"},
		{&arraySetArray, "mlx_array_set"},
		{&opAdd, "mlx_add"},
		{&opMatmul, "mlx_matmul"},
		{&opAstype, "mlx_astype"},
	} {
		if _, err := dl.Sym(lib, b.name); err != nil {
			return fmt.Errorf("%s has no %s: %w", path, b.name, err)
		}
		purego.RegisterLibFunc(b.fn, lib, b.name)
	}
	if err := bindOps(lib); err != nil {
		return err
	}
	if err := bindIO(lib); err != nil {
		return err
	}
	if err := bindKernels(lib); err != nil {
		return err
	}
	if err := bindRandom(lib); err != nil {
		return err
	}
	if err := bindQuant(lib); err != nil {
		return err
	}
	if err := bindMemory(lib); err != nil {
		return err
	}
	bindAudio(lib) // audio_ops.go: best-effort, a library without them serves text
	bindImage(lib) // image_ops.go: the same
	setErrorHandler(purego.NewCallback(func(msg *byte, _ uintptr) {
		lastErr = goString(msg)
	}), 0, 0)
	const mlxCPU, mlxGPU = 0, 1 // enum mlx_device_type
	gpu = streamNewUnsafe(deviceNewType(mlxGPU, 0))
	cpu = streamNewUnsafe(deviceNewType(mlxCPU, 0))
	if gpu == 0 || cpu == 0 {
		return fmt.Errorf("no GPU stream: %s", lastErr)
	}
	return nil
}

func goString(p *byte) string {
	if p == nil {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(p, n))
}

// check panics with MLX's message when a call reports failure.
func check(rc int32) {
	if rc != 0 {
		msg := lastErr
		lastErr = ""
		if msg == "" {
			msg = fmt.Sprintf("call failed (%d) with no message", rc)
		}
		panic(&Error{msg})
	}
}

// Recover turns a panic from a failed op back into an error; call it deferred
// at the top of a step. Other panics pass through.
func Recover(err *error) {
	if r := recover(); r != nil {
		if e, ok := r.(*Error); ok {
			*err = e
			return
		}
		panic(r)
	}
}

// out returns a fresh result slot, owned by the current scope.
func out(rc int32, a Array) Array {
	check(rc)
	track(a)
	return a
}

// FromFloat32 copies data into a new float32 array of the given shape.
func FromFloat32(data []float32, shape ...int) Array {
	return fromData(unsafe.Pointer(unsafe.SliceData(data)), len(data), shape, Float32)
}

// FromInt32 copies data into a new int32 array of the given shape.
func FromInt32(data []int32, shape ...int) Array {
	return fromData(unsafe.Pointer(unsafe.SliceData(data)), len(data), shape, Int32)
}

func fromData(p unsafe.Pointer, n int, shape []int, dt Dtype) Array {
	want := 1
	s32 := make([]int32, len(shape))
	for i, d := range shape {
		want *= d
		s32[i] = int32(d)
	}
	if want != n {
		panic(&Error{fmt.Sprintf("%d values do not fill shape %v", n, shape)})
	}
	a := arrayNewData(p, unsafe.SliceData(s32), int32(len(s32)), dt)
	runtime.KeepAlive(s32)
	if a == 0 {
		check(1)
	}
	track(a)
	return a
}

// Scalar makes a 0-d float32 array.
func Scalar(v float32) Array { a := arrayNewF32(v); track(a); return a }

// Free releases a. Freeing an array a scope also holds is a double free.
func (a Array) Free() { arrayFree(a) }

// Ref is a second handle to a's array, owned by no scope: freeing either
// leaves the other. While both are held MLX will not write an op's result
// into the array in place, so neither handle ever sees the other's updates.
func (a Array) Ref() Array {
	r := arrayNew()
	check(arraySetArray(&r, a))
	return r
}

// Eval runs the graph that produces a.
func (a Array) Eval() { check(arrayEval(a)) }

// Shape is a's dimensions.
func (a Array) Shape() []int {
	n := int(arrayNdim(a))
	if n == 0 {
		return nil
	}
	s := unsafe.Slice(arrayShape(a), n)
	r := make([]int, n)
	for i, d := range s {
		r[i] = int(d)
	}
	return r
}

// Dtype is a's element type.
func (a Array) Dtype() Dtype { return arrayDtype(a) }

// Size is a's element count.
func (a Array) Size() int { return int(arraySize(a)) }

// Nbytes is the size of a's elements in bytes.
func (a Array) Nbytes() int { return int(arrayNbytes(a)) }

// Float32s evaluates a, which must be float32, and copies its elements out.
func (a Array) Float32s() []float32 {
	if a.Dtype() != Float32 {
		panic(&Error{"Float32s of a " + a.Dtype().String() + " array"})
	}
	a.Eval()
	p := arrayDataF32(a)
	if p == nil {
		check(1)
	}
	return append([]float32(nil), unsafe.Slice(p, a.Size())...)
}

// Int32s evaluates a, which must be int32, and copies its elements out.
func (a Array) Int32s() []int32 {
	if a.Dtype() != Int32 {
		panic(&Error{"Int32s of a " + a.Dtype().String() + " array"})
	}
	a.Eval()
	p := arrayDataI32(a)
	if p == nil {
		check(1)
	}
	return append([]int32(nil), unsafe.Slice(p, a.Size())...)
}

// Synchronize waits for everything queued on the GPU stream.
func Synchronize() { check(synchronize(gpu)) }

// Add is a + b, broadcasting.
func Add(a, b Array) Array { r := arrayNew(); return out(opAdd(&r, a, b, gpu), r) }

// Matmul is a @ b, batched over leading dimensions.
func Matmul(a, b Array) Array { r := arrayNew(); return out(opMatmul(&r, a, b, gpu), r) }

// Astype converts a to dt.
func Astype(a Array, dt Dtype) Array { r := arrayNew(); return out(opAstype(&r, a, dt, gpu), r) }
