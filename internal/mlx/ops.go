package mlx

import (
	"fmt"
	"unsafe"

	"github.com/LocalKinAI/kinfer/internal/dl"
	"github.com/ebitengine/purego"
)

// The op surface a transformer needs, bound from mlx-c's ops.h and fast.h.
// Each wrapper makes an empty result, lets MLX fill it, and hands it to the
// current scope; shapes and axes go across as int32 arrays.

type (
	unaryFn  func(res *Array, a Array, s Stream) int32
	binaryFn func(res *Array, a, b Array, s Stream) int32
	vector   uintptr // mlx_vector_array
)

// optFloat mirrors mlx_optional_float (8 bytes).
type optFloat struct {
	Value float32
	Has   uint8
	_     [3]byte
}

var (
	opSigmoid, opExp, opLog, opNegative, opRsqrt, opSquare, opSqrt, opAbs, opCopy, opTransposeAll unaryFn

	opSubtract, opMultiply, opDivide, opFloorDivide, opMaximum, opMinimum, opGreater, opLess, opEqual,
	opLogaddexp, opPower binaryFn

	opReshape       func(res *Array, a Array, shape *int32, n uintptr, s Stream) int32
	opTransposeAxes func(res *Array, a Array, axes *int32, n uintptr, s Stream) int32
	opBroadcastTo   func(res *Array, a Array, shape *int32, n uintptr, s Stream) int32
	opSlice         func(res *Array, a Array, start *int32, ns uintptr, stop *int32, ne uintptr, strides *int32, nt uintptr, s Stream) int32
	opSliceUpdate   func(res *Array, src, upd Array, start *int32, ns uintptr, stop *int32, ne uintptr, strides *int32, nt uintptr, s Stream) int32
	opExpandDims    func(res *Array, a Array, axis int32, s Stream) int32
	opSqueezeAxis   func(res *Array, a Array, axis int32, s Stream) int32
	opSwapaxes      func(res *Array, a Array, a1, a2 int32, s Stream) int32
	opContiguous    func(res *Array, a Array, allowColMajor bool, s Stream) int32
	opRepeatAxis    func(res *Array, a Array, repeats, axis int32, s Stream) int32
	opSplitSections func(res *vector, a Array, idx *int32, n uintptr, axis int32, s Stream) int32
	opConcatenate   func(res *Array, v vector, axis int32, s Stream) int32
	opStack         func(res *Array, v vector, axis int32, s Stream) int32

	opTake          func(res *Array, a, idx Array, s Stream) int32
	opTakeAxis      func(res *Array, a, idx Array, axis int32, s Stream) int32
	opTakeAlongAxis func(res *Array, a, idx Array, axis int32, s Stream) int32
	opPutAlongAxis  func(res *Array, a, idx, vals Array, axis int32, s Stream) int32
	opWhere         func(res *Array, cond, x, y Array, s Stream) int32

	opSumAxis       func(res *Array, a Array, axis int32, keep bool, s Stream) int32
	opMeanAxis      func(res *Array, a Array, axis int32, keep bool, s Stream) int32
	opMaxAxis       func(res *Array, a Array, axis int32, keep bool, s Stream) int32
	opArgmaxAxis    func(res *Array, a Array, axis int32, keep bool, s Stream) int32
	opLogsumexpAxis func(res *Array, a Array, axis int32, keep bool, s Stream) int32
	opSoftmaxAxis   func(res *Array, a Array, axis int32, precise bool, s Stream) int32
	opTopkAxis      func(res *Array, a Array, k, axis int32, s Stream) int32
	opArgpartition  func(res *Array, a Array, kth, axis int32, s Stream) int32
	opSortAxis      func(res *Array, a Array, axis int32, s Stream) int32
	opArgsortAxis   func(res *Array, a Array, axis int32, s Stream) int32

	opZeros  func(res *Array, shape *int32, n uintptr, dt Dtype, s Stream) int32
	opOnes   func(res *Array, shape *int32, n uintptr, dt Dtype, s Stream) int32
	opFull   func(res *Array, shape *int32, n uintptr, v Array, dt Dtype, s Stream) int32
	opArange func(res *Array, start, stop, step float64, dt Dtype, s Stream) int32

	opConv1d func(res *Array, in, w Array, stride, padding, dilation, groups int32, s Stream) int32

	opRMSNorm func(res *Array, x, w Array, eps float32, s Stream) int32
	opRoPE    func(res *Array, x Array, dims int32, traditional bool, base optFloat, scale float32, offset int32, freqs Array, s Stream) int32
	opRoPEAt  func(res *Array, x Array, dims int32, traditional bool, base optFloat, scale float32, offset Array, freqs Array, s Stream) int32
	opSDPA    func(res *Array, q, k, v Array, scale float32, maskMode string, mask, sinks Array, forceFused bool, s Stream) int32

	evalVec      func(vector) int32
	asyncEvalVec func(vector) int32

	vecNew    func() vector
	vecFree   func(vector) int32
	vecAppend func(vector, Array) int32
	vecSize   func(vector) uintptr
	vecGet    func(res *Array, v vector, i uintptr) int32
)

func bindOps(lib uintptr) error {
	for _, b := range []struct {
		fn   any
		name string
	}{
		{&opSigmoid, "mlx_sigmoid"}, {&opExp, "mlx_exp"}, {&opLog, "mlx_log"},
		{&opNegative, "mlx_negative"}, {&opRsqrt, "mlx_rsqrt"}, {&opSquare, "mlx_square"},
		{&opSqrt, "mlx_sqrt"}, {&opAbs, "mlx_abs"}, {&opCopy, "mlx_copy"},
		{&opTransposeAll, "mlx_transpose"},
		{&opSubtract, "mlx_subtract"}, {&opMultiply, "mlx_multiply"}, {&opDivide, "mlx_divide"},
		{&opFloorDivide, "mlx_floor_divide"},
		{&opMaximum, "mlx_maximum"}, {&opMinimum, "mlx_minimum"}, {&opGreater, "mlx_greater"},
		{&opLess, "mlx_less"}, {&opEqual, "mlx_equal"}, {&opLogaddexp, "mlx_logaddexp"},
		{&opPower, "mlx_power"},
		{&opReshape, "mlx_reshape"}, {&opTransposeAxes, "mlx_transpose_axes"},
		{&opBroadcastTo, "mlx_broadcast_to"}, {&opSlice, "mlx_slice"},
		{&opSliceUpdate, "mlx_slice_update"}, {&opExpandDims, "mlx_expand_dims"},
		{&opSqueezeAxis, "mlx_squeeze_axis"}, {&opSwapaxes, "mlx_swapaxes"},
		{&opContiguous, "mlx_contiguous"}, {&opRepeatAxis, "mlx_repeat_axis"},
		{&opSplitSections, "mlx_split_sections"}, {&opConcatenate, "mlx_concatenate_axis"},
		{&opStack, "mlx_stack_axis"},
		{&opTake, "mlx_take"}, {&opTakeAxis, "mlx_take_axis"},
		{&opTakeAlongAxis, "mlx_take_along_axis"}, {&opPutAlongAxis, "mlx_put_along_axis"},
		{&opWhere, "mlx_where"},
		{&opSumAxis, "mlx_sum_axis"}, {&opMeanAxis, "mlx_mean_axis"}, {&opMaxAxis, "mlx_max_axis"},
		{&opArgmaxAxis, "mlx_argmax_axis"}, {&opLogsumexpAxis, "mlx_logsumexp_axis"},
		{&opSoftmaxAxis, "mlx_softmax_axis"}, {&opTopkAxis, "mlx_topk_axis"},
		{&opArgpartition, "mlx_argpartition_axis"}, {&opSortAxis, "mlx_sort_axis"},
		{&opArgsortAxis, "mlx_argsort_axis"},
		{&opZeros, "mlx_zeros"}, {&opOnes, "mlx_ones"}, {&opFull, "mlx_full"},
		{&opArange, "mlx_arange"},
		{&opConv1d, "mlx_conv1d"},
		{&opRMSNorm, "mlx_fast_rms_norm"}, {&opRoPE, "mlx_fast_rope"}, {&opRoPEAt, "mlx_fast_rope_dynamic"},
		{&opSDPA, "mlx_fast_scaled_dot_product_attention"},
		{&evalVec, "mlx_eval"}, {&asyncEvalVec, "mlx_async_eval"},
		{&vecNew, "mlx_vector_array_new"}, {&vecFree, "mlx_vector_array_free"},
		{&vecAppend, "mlx_vector_array_append_value"}, {&vecSize, "mlx_vector_array_size"},
		{&vecGet, "mlx_vector_array_get"},
	} {
		if _, err := dl.Sym(lib, b.name); err != nil {
			return fmt.Errorf("mlx-c has no %s: %w", b.name, err)
		}
		purego.RegisterLibFunc(b.fn, lib, b.name)
	}
	return nil
}

func ints(v []int) (*int32, uintptr) {
	if len(v) == 0 {
		return nil, 0
	}
	r := make([]int32, len(v))
	for i, x := range v {
		r[i] = int32(x)
	}
	return unsafe.SliceData(r), uintptr(len(r))
}

func un(f unaryFn, a Array) Array      { r := arrayNew(); return out(f(&r, a, gpu), r) }
func bin(f binaryFn, a, b Array) Array { r := arrayNew(); return out(f(&r, a, b, gpu), r) }
func Sigmoid(a Array) Array            { return un(opSigmoid, a) }
func Exp(a Array) Array                { return un(opExp, a) }
func Log(a Array) Array                { return un(opLog, a) }
func Negative(a Array) Array           { return un(opNegative, a) }
func Rsqrt(a Array) Array              { return un(opRsqrt, a) }
func Square(a Array) Array             { return un(opSquare, a) }
func Sqrt(a Array) Array               { return un(opSqrt, a) }
func Abs(a Array) Array                { return un(opAbs, a) }
func Copy(a Array) Array               { return un(opCopy, a) }
func T(a Array) Array                  { return un(opTransposeAll, a) }
func Subtract(a, b Array) Array        { return bin(opSubtract, a, b) }
func Multiply(a, b Array) Array        { return bin(opMultiply, a, b) }
func Divide(a, b Array) Array          { return bin(opDivide, a, b) }
func FloorDivide(a, b Array) Array     { return bin(opFloorDivide, a, b) }
func Maximum(a, b Array) Array         { return bin(opMaximum, a, b) }
func Minimum(a, b Array) Array         { return bin(opMinimum, a, b) }
func Greater(a, b Array) Array         { return bin(opGreater, a, b) }
func Less(a, b Array) Array            { return bin(opLess, a, b) }
func Equal(a, b Array) Array           { return bin(opEqual, a, b) }
func Logaddexp(a, b Array) Array       { return bin(opLogaddexp, a, b) }
func Power(a, b Array) Array           { return bin(opPower, a, b) }

// MulScalar and AddScalar keep a's dtype, the way a Python float does in
// mlx: a float32 scalar array would promote a bfloat16 a to float32.
func MulScalar(a Array, v float32) Array { return Multiply(a, ScalarAs(v, a.Dtype())) }
func AddScalar(a Array, v float32) Array { return Add(a, ScalarAs(v, a.Dtype())) }

// ScalarAs is a 0-d array of dtype dt.
func ScalarAs(v float32, dt Dtype) Array {
	s := Scalar(v)
	if dt == Float32 {
		return s
	}
	return Astype(s, dt)
}

// Eval runs the graphs of all the arrays together, so work they share is done
// once.
func Eval(arrays ...Array) {
	v := toVector(arrays)
	defer vecFree(v)
	check(evalVec(v))
}

// AsyncEval starts evaluating the arrays and returns at once; reading one
// waits for it.
func AsyncEval(arrays ...Array) {
	v := toVector(arrays)
	defer vecFree(v)
	check(asyncEvalVec(v))
}

// SiLU is x * sigmoid(x).
func SiLU(x Array) Array { return Multiply(x, Sigmoid(x)) }

// Softplus is log(1 + e^x), as logaddexp(x, 0), in x's dtype.
func Softplus(x Array) Array { return Logaddexp(x, ScalarAs(0, x.Dtype())) }

func Reshape(a Array, shape ...int) Array {
	p, n := ints(shape)
	r := arrayNew()
	return out(opReshape(&r, a, p, n, gpu), r)
}

func Transpose(a Array, axes ...int) Array {
	p, n := ints(axes)
	r := arrayNew()
	return out(opTransposeAxes(&r, a, p, n, gpu), r)
}

func BroadcastTo(a Array, shape ...int) Array {
	p, n := ints(shape)
	r := arrayNew()
	return out(opBroadcastTo(&r, a, p, n, gpu), r)
}

// Slice is a[start:stop] along every axis, step 1.
func Slice(a Array, start, stop []int) Array {
	ps, ns := ints(start)
	pe, ne := ints(stop)
	ones := make([]int, len(start))
	for i := range ones {
		ones[i] = 1
	}
	pt, nt := ints(ones)
	r := arrayNew()
	return out(opSlice(&r, a, ps, ns, pe, ne, pt, nt, gpu), r)
}

// SliceUpdate writes upd into src[start:stop] and returns the result.
func SliceUpdate(src, upd Array, start, stop []int) Array {
	ps, ns := ints(start)
	pe, ne := ints(stop)
	ones := make([]int, len(start))
	for i := range ones {
		ones[i] = 1
	}
	pt, nt := ints(ones)
	r := arrayNew()
	return out(opSliceUpdate(&r, src, upd, ps, ns, pe, ne, pt, nt, gpu), r)
}

func ExpandDims(a Array, axis int) Array {
	r := arrayNew()
	return out(opExpandDims(&r, a, int32(axis), gpu), r)
}

func Squeeze(a Array, axis int) Array {
	r := arrayNew()
	return out(opSqueezeAxis(&r, a, int32(axis), gpu), r)
}

func Swapaxes(a Array, a1, a2 int) Array {
	r := arrayNew()
	return out(opSwapaxes(&r, a, int32(a1), int32(a2), gpu), r)
}

func Contiguous(a Array) Array {
	r := arrayNew()
	return out(opContiguous(&r, a, false, gpu), r)
}

func Repeat(a Array, repeats, axis int) Array {
	r := arrayNew()
	return out(opRepeatAxis(&r, a, int32(repeats), int32(axis), gpu), r)
}

// Split cuts a along axis at the given indices, as mx.split(a, indices, axis).
func Split(a Array, axis int, indices ...int) []Array {
	p, n := ints(indices)
	v := vecNew()
	defer vecFree(v)
	check(opSplitSections(&v, a, p, n, int32(axis), gpu))
	return fromVector(v)
}

func fromVector(v vector) []Array {
	res := make([]Array, vecSize(v))
	for i := range res {
		r := arrayNew()
		check(vecGet(&r, v, uintptr(i)))
		track(r)
		res[i] = r
	}
	return res
}

func toVector(arrays []Array) vector {
	v := vecNew()
	for _, a := range arrays {
		check(vecAppend(v, a))
	}
	return v
}

func Concatenate(axis int, arrays ...Array) Array {
	v := toVector(arrays)
	defer vecFree(v)
	r := arrayNew()
	return out(opConcatenate(&r, v, int32(axis), gpu), r)
}

func Stack(axis int, arrays ...Array) Array {
	v := toVector(arrays)
	defer vecFree(v)
	r := arrayNew()
	return out(opStack(&r, v, int32(axis), gpu), r)
}

// Take gathers rows of a (axis 0) at idx — an embedding lookup.
func Take(a, idx Array, axis int) Array {
	r := arrayNew()
	return out(opTakeAxis(&r, a, idx, int32(axis), gpu), r)
}

func TakeAlongAxis(a, idx Array, axis int) Array {
	r := arrayNew()
	return out(opTakeAlongAxis(&r, a, idx, int32(axis), gpu), r)
}

func PutAlongAxis(a, idx, vals Array, axis int) Array {
	r := arrayNew()
	return out(opPutAlongAxis(&r, a, idx, vals, int32(axis), gpu), r)
}

func Where(cond, x, y Array) Array {
	r := arrayNew()
	return out(opWhere(&r, cond, x, y, gpu), r)
}

func reduce(f func(res *Array, a Array, axis int32, keep bool, s Stream) int32, a Array, axis int, keep bool) Array {
	r := arrayNew()
	return out(f(&r, a, int32(axis), keep, gpu), r)
}

func Sum(a Array, axis int, keep bool) Array       { return reduce(opSumAxis, a, axis, keep) }
func Mean(a Array, axis int, keep bool) Array      { return reduce(opMeanAxis, a, axis, keep) }
func Max(a Array, axis int, keep bool) Array       { return reduce(opMaxAxis, a, axis, keep) }
func Argmax(a Array, axis int, keep bool) Array    { return reduce(opArgmaxAxis, a, axis, keep) }
func Logsumexp(a Array, axis int, keep bool) Array { return reduce(opLogsumexpAxis, a, axis, keep) }

func Softmax(a Array, axis int) Array {
	r := arrayNew()
	return out(opSoftmaxAxis(&r, a, int32(axis), true, gpu), r)
}

func Topk(a Array, k, axis int) Array {
	r := arrayNew()
	return out(opTopkAxis(&r, a, int32(k), int32(axis), gpu), r)
}

func Argpartition(a Array, kth, axis int) Array {
	r := arrayNew()
	return out(opArgpartition(&r, a, int32(kth), int32(axis), gpu), r)
}

func Sort(a Array, axis int) Array {
	r := arrayNew()
	return out(opSortAxis(&r, a, int32(axis), gpu), r)
}

func Argsort(a Array, axis int) Array {
	r := arrayNew()
	return out(opArgsortAxis(&r, a, int32(axis), gpu), r)
}

func Zeros(dt Dtype, shape ...int) Array {
	p, n := ints(shape)
	r := arrayNew()
	return out(opZeros(&r, p, n, dt, gpu), r)
}

func Ones(dt Dtype, shape ...int) Array {
	p, n := ints(shape)
	r := arrayNew()
	return out(opOnes(&r, p, n, dt, gpu), r)
}

func Full(dt Dtype, v float32, shape ...int) Array {
	p, n := ints(shape)
	r := arrayNew()
	return out(opFull(&r, p, n, Scalar(v), dt, gpu), r)
}

func Arange(start, stop, step float64, dt Dtype) Array {
	r := arrayNew()
	return out(opArange(&r, start, stop, step, dt, gpu), r)
}

// Conv1d convolves in [N, L, C] with w [Cout, K, C/groups], as mx.conv1d.
func Conv1d(in, w Array, stride, padding, dilation, groups int) Array {
	r := arrayNew()
	return out(opConv1d(&r, in, w, int32(stride), int32(padding), int32(dilation), int32(groups), gpu), r)
}

// RMSNorm normalizes the last axis; w may be 0 for no weight.
func RMSNorm(x, w Array, eps float32) Array {
	r := arrayNew()
	return out(opRMSNorm(&r, x, w, eps, gpu), r)
}

// RoPE rotates the first dims features of the last axis of x [B, H, L, D],
// positions starting at offset.
func RoPE(x Array, dims int, traditional bool, base, scale float32, offset int) Array {
	r := arrayNew()
	return out(opRoPE(&r, x, int32(dims), traditional, optFloat{Value: base, Has: 1}, scale, int32(offset), 0, gpu), r)
}

// RoPEAt is RoPE with one starting position per batch row: offsets [B]
// int32, for x [B, H, L, D].
func RoPEAt(x Array, dims int, traditional bool, base, scale float32, offsets Array) Array {
	r := arrayNew()
	return out(opRoPEAt(&r, x, int32(dims), traditional, optFloat{Value: base, Has: 1}, scale, offsets, 0, gpu), r)
}

// SDPA is fused attention over q [B, Hq, L, D] and k, v [B, Hkv, S, D]. mode
// is "" (mask may then be an array or 0) or "causal".
func SDPA(q, k, v Array, scale float32, mode string, mask Array) Array {
	r := arrayNew()
	return out(opSDPA(&r, q, k, v, scale, mode, mask, 0, false, gpu), r)
}

// optInt mirrors mlx_optional_int (8 bytes).
type optInt struct {
	Value int32
	Has   uint8
	_     [3]byte
}

var (
	opQuantizedMatmul func(res *Array, x, w, scales, biases Array, transpose bool, group, bits optInt, mode string, s Stream) int32
	opDequantize      func(res *Array, w, scales, biases Array, group, bits optInt, mode string, globalScale Array, dt optDtype, s Stream) int32
	opGatherQMM       func(res *Array, x, w, scales, biases, lhs, rhs Array, transpose bool, group, bits optInt, mode string, globalScale Array, sorted bool, s Stream) int32
	opGatherMM        func(res *Array, a, b, lhs, rhs Array, sorted bool, s Stream) int32
)

func bindQuant(lib uintptr) error {
	for _, b := range []struct {
		fn   any
		name string
	}{
		{&opQuantizedMatmul, "mlx_quantized_matmul"}, {&opDequantize, "mlx_dequantize"},
		{&opGatherQMM, "mlx_gather_qmm"}, {&opGatherMM, "mlx_gather_mm"},
	} {
		if _, err := dl.Sym(lib, b.name); err != nil {
			return fmt.Errorf("mlx-c has no %s: %w", b.name, err)
		}
		purego.RegisterLibFunc(b.fn, lib, b.name)
	}
	return nil
}

// QuantizedMatmul is x @ dequantize(w).T for w quantized as MLX stores a
// QuantizedLinear: packed weights, per-group scales and (affine) biases.
func QuantizedMatmul(x, w, scales, biases Array, group, bits int, mode string) Array {
	r := arrayNew()
	return out(opQuantizedMatmul(&r, x, w, scales, biases, true,
		optInt{Value: int32(group), Has: 1}, optInt{Value: int32(bits), Has: 1}, mode, gpu), r)
}

// GatherQMM is QuantizedMatmul with a stack of weights, w [E, out, in/…]: each
// of x's matrices is multiplied by the weight rhs names for it, x [..., M, in]
// broadcast against rhs [...] to [..., M, out]. sorted says rhs is in
// ascending order, which lets MLX read each weight once for a run of rows.
func GatherQMM(x, w, scales, biases, rhs Array, group, bits int, mode string, sorted bool) Array {
	r := arrayNew()
	return out(opGatherQMM(&r, x, w, scales, biases, 0, rhs, true,
		optInt{Value: int32(group), Has: 1}, optInt{Value: int32(bits), Has: 1}, mode, 0, sorted, gpu), r)
}

// GatherMM is a @ b[rhs] for a stack of matrices b [E, K, N], as GatherQMM
// for weights that are not quantized; b is not transposed.
func GatherMM(a, b, rhs Array, sorted bool) Array {
	r := arrayNew()
	return out(opGatherMM(&r, a, b, 0, rhs, sorted, gpu), r)
}

// Dequantize unpacks quantized w to the scales' dtype.
func Dequantize(w, scales, biases Array, group, bits int, mode string) Array {
	r := arrayNew()
	return out(opDequantize(&r, w, scales, biases, optInt{Value: int32(group), Has: 1},
		optInt{Value: int32(bits), Has: 1}, mode, 0, optDtype{}, gpu), r)
}
