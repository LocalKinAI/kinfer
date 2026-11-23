package mlxmodel

import (
	"fmt"
	"sync"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// A decode step of a large model is a few thousand operations, and all but a
// few hundred of them are small: a scale, a cast, a sigmoid over a few
// thousand numbers. Each still costs the GPU four to five microseconds to
// take up — measured on an M3 Ultra, a thousand of them are 4 ms of a 35 ms
// step of Qwen3.8-27B, whatever the weights cost to read. The kernels here do
// a run of such operations in one: the same arithmetic in the same order and
// the same types, rounding where the operations they stand for round, so a
// model computes exactly what it did (TestFusedKernelsMatchOps,
// TestQwen35FusedMatchesOps).
//
// opsSmall, which the tests set, has the model run the operations themselves:
// the reference the kernels are tested against.
var opsSmall bool

// sigmoidOf is MLX's Sigmoid (mlx/backend/metal/kernels/unary_ops.h, MIT,
// Apple Inc.) for a variable x of type T, giving s.
func sigmoidOf(T, x, s string) string {
	return fmt.Sprintf(`
        %[1]s %[3]s;
        {
          auto y_ = 1 / (1 + metal::precise::exp(metal::abs(%[2]s)));
          %[3]s = (%[2]s < 0) ? y_ : 1 - y_;
        }`, T, x, s)
}

// scaled is a product with a quantized weight, y, times the weight's own
// scale s in float32 and rounded back to y's type: Linear.global.
func scaled(y, s string) string {
	return fmt.Sprintf("static_cast<InT>(static_cast<float>(%s) * %s)", y, s)
}

var fusedSources = map[string]struct {
	in, out []string
	src     string
}{
	// out = y·s
	"kinfer_scale": {[]string{"y", "s"}, []string{"out"}, `
        uint i = thread_position_in_grid.x;
        out[i] = ` + scaled("y[i]", "s") + `;
`},
	// out = h + y·s
	"kinfer_add_scaled": {[]string{"h", "y", "s"}, []string{"out"}, `
        uint i = thread_position_in_grid.x;
        InT r = ` + scaled("y[i]", "s") + `;
        out[i] = h[i] + r;
`},
	// out = silu(g·sg) · (u·su)
	"kinfer_swiglu": {[]string{"g", "sg", "u", "su"}, []string{"out"}, `
        uint i = thread_position_in_grid.x;
        InT gs = ` + scaled("g[i]", "sg") + `;
        InT us = ` + scaled("u[i]", "su") + `;` + sigmoidOf("InT", "gs", "sig") + `
        InT act = gs * sig;
        out[i] = act * us;
`},
	// out = o · sigmoid(gate)
	"kinfer_gate": {[]string{"o", "gate"}, []string{"out"}, `
        uint i = thread_position_in_grid.x;
        InT gt = gate[i];` + sigmoidOf("InT", "gt", "sig") + `
        out[i] = o[i] * sig;
`},
	// A DeltaNet layer's gates, per head: g = exp(nexp · softplus(a·sa + dt))
	// in float32 and beta = sigmoid(b·sb). The softplus is MLX's LogAddExp
	// with zero (binary_ops.h), in the type the sum has.
	"kinfer_delta_gates": {[]string{"a", "sa", "b", "sb", "nexp", "dt"}, []string{"g", "beta"}, `
        uint i = thread_position_in_grid.x;
        uint hd = i % Hv;
        InT av = ` + scaled("a[i]", "sa") + `;
        SumT x = static_cast<SumT>(av) + static_cast<SumT>(dt[hd]);
        SumT zero = static_cast<SumT>(0);
        SumT sp;
        if (metal::isnan(x) || metal::isnan(zero)) {
          sp = metal::numeric_limits<SumT>::quiet_NaN();
        } else {
          constexpr SumT inf = metal::numeric_limits<SumT>::infinity();
          SumT maxval = metal::max(x, zero);
          SumT minval = metal::min(x, zero);
          sp = (minval == -inf || maxval == inf)
              ? maxval
              : (maxval + log1p(metal::exp(minval - maxval)));
        }
        float e = nexp[hd] * static_cast<float>(sp);
        g[i] = metal::precise::exp(e);
        InT bs = ` + scaled("b[i]", "sb") + `;` + sigmoidOf("InT", "bs", "sig") + `
        beta[i] = sig;
`},
	// A DeltaNet layer's output: rms_norm(y), given as n, times silu(z·sz),
	// the product in float32.
	"kinfer_delta_out": {[]string{"z", "sz", "n"}, []string{"out"}, `
        uint i = thread_position_in_grid.x;
        InT zs = ` + scaled("z[i]", "sz") + `;
        float zf = static_cast<float>(zs);` + sigmoidOf("float", "zf", "sig") + `
        float act = zf * sig;
        float nf = static_cast<float>(n[i]);
        out[i] = static_cast<InT>(act * nf);
`},
}

var (
	fusedMu      sync.Mutex
	fusedKernels = map[string]*mlx.Kernel{}
	fusedLaunch  = map[string]*mlx.Launch{}
)

// fused runs the named kernel over arrays of n elements. Its outputs have the
// given shapes and types; tpl are its template arguments beyond InT, the
// type of the activations. A launch is set up once for each shape it meets.
func fused(name string, inT mlx.Dtype, n int, outs []mlx.KernelOutput, tpl []mlx.KernelTemplate, inputs ...mlx.Array) []mlx.Array {
	key := fmt.Sprint(name, inT, outs, tpl)
	fusedMu.Lock()
	l := fusedLaunch[key]
	if l == nil {
		k := fusedKernels[name]
		if k == nil {
			s := fusedSources[name]
			k = mlx.NewKernel(name, s.in, s.out, s.src)
			fusedKernels[name] = k
		}
		// A launch for every length of prompt would be kept for good; only
		// the shapes a step or a pass that checks drafts has are.
		l = k.Prepare(outs, append([]mlx.KernelTemplate{{Name: "InT", Dtype: inT}}, tpl...), [3]int{n, 1, 1}, [3]int{256, 1, 1})
		if len(fusedLaunch) < 4096 {
			fusedLaunch[key] = l
		} else {
			defer l.Free()
		}
	}
	fusedMu.Unlock()
	return l.Run(inputs...)
}

func size(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// scale is the weight's own scale as a kernel takes it: one where it has
// none, which leaves a product as it is.
func (l Linear) scale() mlx.Array {
	if l.Global == 0 || !l.quantized() {
		return mlx.Const(1, mlx.Float32)
	}
	return mlx.Const(l.Global, mlx.Float32)
}

// scaledBy is y, a product with the weight, as linear would have given it.
func (l Linear) scaledBy(y mlx.Array) mlx.Array {
	if !l.quantized() {
		return y
	}
	return l.global(y)
}

// product is x @ w.T before the weight's own scale.
func product(x mlx.Array, w Linear) mlx.Array {
	if w.quantized() {
		return mlx.QuantizedMatmul(x, w.W, w.Scales, w.Biases, w.Group, w.Bits, w.Mode)
	}
	return mlx.Matmul(x, mlx.T(w.W))
}

// halfOrFloat reports whether dt is a type the kernels are written for.
func halfOrFloat(dt mlx.Dtype) bool {
	return dt == mlx.Bfloat16 || dt == mlx.Float16 || dt == mlx.Float32
}

// project is linear for a layer's own projections: the scale in one
// operation where linear's takes three.
func project(x mlx.Array, w Linear) mlx.Array {
	if opsSmall || w.Global == 0 || !w.quantized() {
		return linear(x, w)
	}
	y := product(x, w)
	if !halfOrFloat(y.Dtype()) {
		return w.scaledBy(y)
	}
	sh := y.Shape()
	return fused("kinfer_scale", y.Dtype(), size(sh), []mlx.KernelOutput{{Shape: sh, Dtype: y.Dtype()}}, nil, y, w.scale())[0]
}

// addProjected is h + linear(x, w).
func addProjected(h, x mlx.Array, w Linear) mlx.Array {
	if opsSmall {
		return mlx.Add(h, linear(x, w))
	}
	y := product(x, w)
	sh := y.Shape()
	if y.Dtype() != h.Dtype() || !halfOrFloat(y.Dtype()) || size(h.Shape()) != size(sh) {
		return mlx.Add(h, w.scaledBy(y))
	}
	return fused("kinfer_add_scaled", y.Dtype(), size(sh), []mlx.KernelOutput{{Shape: h.Shape(), Dtype: y.Dtype()}}, nil, h, y, w.scale())[0]
}

// swiGLUIn is the part of swiGLU before its last projection:
// silu(gate(x)) · up(x).
func swiGLUIn(x mlx.Array, gate, up Linear) mlx.Array {
	if opsSmall {
		return mlx.Multiply(mlx.SiLU(linear(x, gate)), linear(x, up))
	}
	g, u := product(x, gate), product(x, up)
	if g.Dtype() != u.Dtype() || !halfOrFloat(g.Dtype()) {
		return mlx.Multiply(mlx.SiLU(gate.scaledBy(g)), up.scaledBy(u))
	}
	sh := g.Shape()
	return fused("kinfer_swiglu", g.Dtype(), size(sh), []mlx.KernelOutput{{Shape: sh, Dtype: g.Dtype()}}, nil, g, gate.scale(), u, up.scale())[0]
}

// gated is o · sigmoid(gate), an attention layer's output before its last
// projection.
func gated(o, gate mlx.Array) mlx.Array {
	if opsSmall || o.Dtype() != gate.Dtype() || !halfOrFloat(o.Dtype()) || size(o.Shape()) != size(gate.Shape()) {
		return mlx.Multiply(o, mlx.Sigmoid(gate))
	}
	sh := o.Shape()
	return fused("kinfer_gate", o.Dtype(), size(sh), []mlx.KernelOutput{{Shape: sh, Dtype: o.Dtype()}}, nil, o, gate)[0]
}

// gateInputs are a DeltaNet layer's projections of x for its output gate, z,
// and its write strength and decay, b and a: each as linear gives it when the
// operations themselves are run, and before the weight's own scale otherwise,
// which deltaGates and deltaOut apply.
func (w *linearAttn) gateInputs(x mlx.Array) (z, b, a mlx.Array) {
	if opsSmall {
		return linear(x, w.inZ), linear(x, w.inB), linear(x, w.inA)
	}
	return product(x, w.inZ), product(x, w.inB), product(x, w.inA)
}

// deltaGates are a DeltaNet layer's decay g, float32, and write strength
// beta, each [B, L, Hv], from gateInputs' a and b.
func (w *linearAttn) deltaGates(a, b mlx.Array) (g, beta mlx.Array) {
	if !opsSmall {
		sum := a.Dtype()
		if w.dtBias.Dtype() != sum {
			sum = mlx.Float32
		}
		dt := w.dtBias.Dtype()
		if a.Dtype() == b.Dtype() && (a.Dtype() == mlx.Bfloat16 || a.Dtype() == mlx.Float32) && (dt == a.Dtype() || dt == mlx.Float32) {
			sh := a.Shape()
			r := fused("kinfer_delta_gates", a.Dtype(), size(sh),
				[]mlx.KernelOutput{{Shape: sh, Dtype: mlx.Float32}, {Shape: sh, Dtype: a.Dtype()}},
				[]mlx.KernelTemplate{{Name: "SumT", Dtype: sum}, {Name: "Hv", Int: sh[len(sh)-1], IsInt: true}},
				a, w.inA.scale(), b, w.inB.scale(), w.nexp, w.dtBias)
			return r[0], r[1]
		}
		a, b = w.inA.scaledBy(a), w.inB.scaledBy(b)
		return mlx.Exp(mlx.Multiply(w.nexp, mlx.Softplus(mlx.Add(a, w.dtBias)))), mlx.Sigmoid(b)
	}
	return mlx.Exp(mlx.Multiply(mlx.Negative(mlx.Exp(mlx.Astype(w.aLog, mlx.Float32))), mlx.Softplus(mlx.Add(a, w.dtBias)))), mlx.Sigmoid(b)
}

// deltaOut is a DeltaNet layer's RMSNormGated: n, the normed y, times
// silu(z), gateInputs' z, the product in float32 and the result of type dt,
// y's own — which n's need not be: a norm's weight is of whatever type the
// model's file has it in, and float32 there makes n float32.
func (w *linearAttn) deltaOut(n, z mlx.Array, dt mlx.Dtype) mlx.Array {
	sh := n.Shape()
	if !opsSmall {
		if z.Dtype() == dt && (dt == mlx.Bfloat16 || dt == mlx.Float32) && halfOrFloat(n.Dtype()) && size(z.Shape()) == size(sh) {
			return fused("kinfer_delta_out", dt, size(sh), []mlx.KernelOutput{{Shape: sh, Dtype: dt}}, nil, z, w.inZ.scale(), n)[0]
		}
		z = w.inZ.scaledBy(z)
	}
	z = mlx.Reshape(z, sh...)
	return mlx.Astype(mlx.Multiply(mlx.SiLU(mlx.Astype(z, mlx.Float32)), mlx.Astype(n, mlx.Float32)), dt)
}
