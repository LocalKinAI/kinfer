package mlxmodel

import (
	"math"
	"sync"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// An Inkling 2 step written out in MLX's own operations is some 19,000 of
// them — 48 layer applications of four hundred — where a step of Qwen3.8-27B
// was 3,400 before its small ones were put together (fused.go), and each
// costs the host a microsecond to record and the GPU several to take up
// whatever it computes. The kernels here each do one part of a layer in one
// operation: the same arithmetic as the operations in inkling2.go, which
// inkPlain has the model use instead and the tests compare them with.
var inkPlain bool

func init() {
	for name, k := range inkSources {
		fusedSources[name] = k
	}
}

// inkWindow reads element j of the convolution's window for row b, channel c:
// its history, then the tokens of this pass.
const inkWindow = `
#define INK_WIN(b, j, c) (((j) < TAIL) ? static_cast<float>(hist[((b) * TAIL + (j)) * NC + (c)]) : static_cast<float>(x[((b) * NL + (j) - TAIL) * NC + (c)]))
        uint i = thread_position_in_grid.x;
        uint c = i % NC;
        uint row = (i / NC) % (NL + TAIL);
        uint b = i / (NC * (NL + TAIL));
`

// inkNext writes the history the pass leaves: the window's last TAIL rows.
const inkNext = `
        } else {
          uint j = row - NL;
          uint from = NL + j;
          if (from < TAIL) {
            next[(b * TAIL + j) * NC + c] = hist[(b * TAIL + from) * NC + c];
          } else {
            next[(b * TAIL + j) * NC + c] = x[(b * NL + from - TAIL) * NC + c];
          }
        }
#undef INK_WIN
`

var inkSources = map[string]struct {
	in, out []string
	src     string
}{
	// inkConv.run with fixed taps: w [TAPS, NC], oldest first.
	// With ADD the result, rounded as it is without, is added to prev in
	// float32 and written in OutT: the layer's residual sum.
	"kinfer_ink_conv": {[]string{"hist", "x", "w", "inv", "prev"}, []string{"out", "next"}, inkWindow + `
        if (row < NL) {
          float sum = static_cast<float>(x[(b * NL + row) * NC + c]);
          for (uint t = 0; t < TAPS; ++t) {
            sum = sum + INK_WIN(b, row + t, c) * w[t * NC + c];
          }
          uint at = (b * NL + row) * NC + c;
          InT y = static_cast<InT>(sum * inv);
          if (ADD) {
            out[at] = static_cast<OutT>(static_cast<float>(prev[at]) + static_cast<float>(y));
          } else {
            out[at] = static_cast<OutT>(y);
          }
` + inkNext},
	// inkConv.run with taps a projection adds to: w [TAPS, NC] by how far back
	// each reaches, shifts [B, NL, HEADS, TAPS].
	"kinfer_ink_conv_taps": {[]string{"hist", "x", "w", "shifts", "inv"}, []string{"out", "next"}, inkWindow + `
        if (row < NL) {
          uint head = c / (NC / HEADS);
          float sum = static_cast<float>(x[(b * NL + row) * NC + c]);
          for (uint t = 0; t < TAPS; ++t) {
            float coef = w[t * NC + c] + static_cast<float>(shifts[((b * NL + row) * HEADS + head) * TAPS + t]);
            sum = sum + INK_WIN(b, row + TAIL - t, c) * coef;
          }
          out[(b * NL + row) * NC + c] = static_cast<InT>(sum * inv);
` + inkNext},
	// inkClip: each row of x [rows, ND], times w [NH, ND] for its head when
	// HASW, shortened to limit, times tau [NL] for its token when HAST.
	"kinfer_ink_clip": {[]string{"x", "w", "tau", "limit"}, []string{"out"}, `
        uint row = thread_position_in_grid.x;
        uint h = row % NH;
        uint l = (row / NH) % NL;
        float sq = 0.0f;
        for (uint d = 0; d < ND; ++d) {
          float v = static_cast<float>(x[row * ND + d]);
          if (HASW) v = v * w[h * ND + d];
          sq += v * v;
        }
        float f = metal::min(limit / (metal::precise::sqrt(sq) + 1e-6f), 1.0f);
        for (uint d = 0; d < ND; ++d) {
          float v = static_cast<float>(x[row * ND + d]);
          if (HASW) v = v * w[h * ND + d];
          v = v * f;
          if (HAST) v = v * tau[l];
          out[row * ND + d] = static_cast<InT>(v);
        }
`},
	// The router's choice for each token: x [N, NHID] times the router's rows
	// [NE, NHID] in float32, and of those logits plus bias [NE] after a
	// sigmoid, the NK largest, in order. A threadgroup of NE threads takes a
	// token: each works out its expert's logit, and after a barrier counts
	// the experts ahead of its own — one thread choosing them one after
	// another took half a millisecond — and the chosen write their place and
	// the sigmoid of their logit, which their shares are in proportion to.
	"kinfer_ink_route": {[]string{"x", "router", "bias"}, []string{"idx", "sig"}, `
        uint e = thread_position_in_threadgroup.x;
        uint n = thread_position_in_grid.z;
        threadgroup float lg[NE];
        float dot = 0.0f;
        for (uint c = 0; c < NHID; ++c) {
          dot += static_cast<float>(x[n * NHID + c]) * router[e * NHID + c];
        }
        float y = 1 / (1 + metal::precise::exp(metal::abs(dot)));
        float sg = (dot < 0) ? y : 1 - y;
        float s = sg + (HASB ? bias[e] : 0.0f);
        lg[e] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint ahead = 0;
        for (uint j = 0; j < NE; ++j) {
          float sj = lg[j];
          ahead += (sj > s || (sj == s && j < e)) ? 1 : 0;
        }
        if (ahead < NK) {
          idx[n * NK + ahead] = e;
          sig[n * NK + ahead] = sg;
        }
`},
	// What an expert's first projection, pair [N*NK, 2*NW] with gate and up
	// by turns, is made into: softcap(gate)·sigmoid(gate)·softcap(up), times
	// the token's share for the expert — its sigmoid's part of the chosen
	// ones' sum, which is the softmax of their log sigmoids, times scale —
	// the projection scaled by the expert's own scale first when HASO.
	"kinfer_ink_act": {[]string{"pair", "idx", "outer", "sig", "scale", "gcap", "ucap"}, []string{"out"}, `
        uint i = thread_position_in_grid.x;
        uint nk = i / NW;
        uint n = nk / NK;
        float all = 0.0f;
        for (uint k = 0; k < NK; ++k) all += sig[n * NK + k];
        float share = sig[nk] / all * scale;
        float o = HASO ? outer[idx[nk]] : 1.0f;
        float g = static_cast<float>(pair[2 * i]) * o;
        float u = static_cast<float>(pair[2 * i + 1]) * o;
        float y = 1 / (1 + metal::precise::exp(metal::abs(g)));
        float sgm = (g < 0) ? y : 1 - y;
        float a = (metal::precise::tanh(g / gcap) * gcap) * sgm * (metal::precise::tanh(u / ucap) * ucap);
        out[i] = static_cast<InT>(a * share);
`},
	// The experts' outputs, down [N, NK, NC], each scaled by its expert's own
	// scale when HASO, summed for the token.
	"kinfer_ink_sum": {[]string{"down", "idx", "outer"}, []string{"out"}, `
        uint i = thread_position_in_grid.x;
        uint c = i % NC;
        uint n = i / NC;
        float sum = 0.0f;
        for (uint k = 0; k < NK; ++k) {
          float o = HASO ? outer[idx[n * NK + k]] : 1.0f;
          sum += static_cast<float>(down[(n * NK + k) * NC + c]) * o;
        }
        out[i] = static_cast<InT>(sum);
`},
	// inkGate.apply's end: input·(1 − tanh(forget)) + message·tanh(gain), gf
	// [rows, 2*NC] holding gain then forget.
	"kinfer_ink_gate": {[]string{"input", "message", "gf"}, []string{"out"}, `
        uint i = thread_position_in_grid.x;
        uint c = i % NC;
        uint row = i / NC;
        float gain = metal::precise::tanh(gf[row * 2 * NC + c]);
        float forget = metal::precise::tanh(gf[row * 2 * NC + NC + c]);
        out[i] = static_cast<float>(input[i]) * (1.0f - forget) + static_cast<float>(message[i]) * gain;
`},
}

// A kernel launched with its outputs, template arguments and grid given anew
// costs the host some twenty calls into MLX before the launch itself; a step
// has a thousand launches. prepared keeps a launch set up for each kernel
// and each set of sizes it has met — id, the numbers the set-up depends on —
// and setup is asked for one only the first time.
type launchKey struct {
	k  *mlx.Kernel
	id [10]int
}

var (
	launchMu sync.Mutex
	launches = map[launchKey]*mlx.Launch{}
)

func prepared(k *mlx.Kernel, id [10]int, setup func() (outs []mlx.KernelOutput, tpl []mlx.KernelTemplate, grid, group [3]int)) *mlx.Launch {
	key := launchKey{k, id}
	launchMu.Lock()
	defer launchMu.Unlock()
	l := launches[key]
	if l == nil {
		outs, tpl, grid, group := setup()
		l = k.Prepare(outs, tpl, grid, group)
		// A launch for every length of prompt would be kept for good; the
		// sizes a step has are few.
		if len(launches) < 4096 {
			launches[key] = l
		}
	}
	return l
}

func tplInt(name string, v int) mlx.KernelTemplate {
	return mlx.KernelTemplate{Name: name, Int: v, IsInt: true}
}

func flag(on bool) int {
	if on {
		return 1
	}
	return 0
}

// inkOne is a one-element array for a kernel input a kernel's flags say it
// does not read: the kernel indexes it all the same.
var inkOne mlx.Array

func one() mlx.Array {
	if inkOne == 0 {
		inkOne = mlx.FromFloat32([]float32{1}, 1)
		mlx.Outlive(inkOne)
	}
	return inkOne
}

func (cv inkConv) fused(x, shifts mlx.Array, hist *mlx.Array) mlx.Array {
	return cv.fusedAdd(x, shifts, hist, 0, x.Dtype())
}

// fusedAdd is fused and, with prev, the sum of prev and the convolution in
// float32, written in the type out: a layer's residual stream after it.
func (cv inkConv) fusedAdd(x, shifts mlx.Array, hist *mlx.Array, prev mlx.Array, out mlx.Dtype) mlx.Array {
	s := x.Shape()
	B, L, C := s[0], s[1], s[2]
	taps := len(cv.rows)
	tail := taps - 1
	h := *hist
	if h == 0 {
		h = mlx.Zeros(x.Dtype(), B, tail, C)
	}
	terms := taps + 1
	name, p := "kinfer_ink_conv", prev
	if shifts != 0 {
		name = "kinfer_ink_conv_taps"
		terms += taps
	} else if p == 0 {
		p = one()
	}
	l := prepared(fusedKernel(name), [10]int{B, L, C, taps, cv.heads, flag(prev != 0), int(out), int(x.Dtype())},
		func() ([]mlx.KernelOutput, []mlx.KernelTemplate, [3]int, [3]int) {
			tpl := []mlx.KernelTemplate{{Name: "InT", Dtype: x.Dtype()}, tplInt("TAPS", taps), tplInt("TAIL", tail), tplInt("NC", C), tplInt("NL", L)}
			if shifts != 0 {
				tpl = append(tpl, tplInt("HEADS", cv.heads))
			} else {
				tpl = append(tpl, mlx.KernelTemplate{Name: "OutT", Dtype: out}, tplInt("ADD", flag(prev != 0)))
			}
			return []mlx.KernelOutput{{Shape: []int{B, L, C}, Dtype: out}, {Shape: []int{B, tail, C}, Dtype: x.Dtype()}},
				tpl, [3]int{B * (L + tail) * C, 1, 1}, [3]int{256, 1, 1}
		})
	inv := mlx.Const(float32(1/math.Sqrt(float64(terms))), mlx.Float32)
	var res []mlx.Array
	if shifts != 0 {
		res = l.Run(h, x, cv.taps, shifts, inv)
	} else {
		res = l.Run(h, x, cv.taps, inv, p)
	}
	*hist = res[1]
	return res[0]
}

// inkClipFused is inkClip for x [B, L, H, D]; tau is [L] float32 or 0.
func inkClipFused(x, weight, tau mlx.Array, limit float32) mlx.Array {
	s := x.Shape()
	B, L, H, D := s[0], s[1], s[2], s[3]
	w, t := weight, tau
	if w == 0 {
		w = one()
	}
	if t == 0 {
		t = one()
	}
	return fused("kinfer_ink_clip", x.Dtype(), B*L*H, []mlx.KernelOutput{{Shape: s, Dtype: x.Dtype()}},
		[]mlx.KernelTemplate{tplInt("NH", H), tplInt("NL", L), tplInt("ND", D), tplInt("HASW", flag(weight != 0)), tplInt("HAST", flag(tau != 0))},
		x, w, t, mlx.Const(limit, mlx.Float32))[0]
}

// routeFused is the router's choice of experts for x [N, hidden]: which, [N,
// K] uint32, and the sigmoid of each one's logit, [N, K] float32.
func (m *Inkling2) routeFused(l *inkLayer, x mlx.Array) (idx, sig mlx.Array) {
	N := x.Shape()[0]
	bias := l.bias
	if bias == 0 {
		bias = one()
	}
	res := prepared(fusedKernel("kinfer_ink_route"), [10]int{N, m.experts, m.topK, m.hidden, flag(l.bias != 0)},
		func() ([]mlx.KernelOutput, []mlx.KernelTemplate, [3]int, [3]int) {
			return []mlx.KernelOutput{{Shape: []int{N, m.topK}, Dtype: mlx.Uint32}, {Shape: []int{N, m.topK}, Dtype: mlx.Float32}},
				[]mlx.KernelTemplate{tplInt("NE", m.experts), tplInt("NK", m.topK), tplInt("NHID", m.hidden), tplInt("HASB", flag(l.bias != 0))},
				[3]int{m.experts, 1, N}, [3]int{m.experts, 1, 1}
		}).Run(x, l.routerRows, bias)
	return res[0], res[1]
}

// fusedKernel is the kernel of that name among inkSources, made once.
func fusedKernel(name string) *mlx.Kernel {
	fusedMu.Lock()
	defer fusedMu.Unlock()
	k := fusedKernels[name]
	if k == nil {
		src := fusedSources[name]
		k = mlx.NewKernel(name, src.in, src.out, src.src)
		fusedKernels[name] = k
	}
	return k
}

// feedForwardFused is feedForward with the routing, the activation and the
// sum each one operation.
func (m *Inkling2) feedForwardFused(l *inkLayer, x mlx.Array) mlx.Array {
	s := x.Shape()
	B, L, H := s[0], s[1], s[2]
	N, K, W := B*L, m.topK, m.width
	flat := mlx.Reshape(x, N, H)
	idx, sig := m.routeFused(l, flat)
	if N*K >= sortAt {
		return m.feedForwardSorted(l, x, flat, idx, sig)
	}

	outer := func(b inkBank) mlx.Array {
		if b.outer == 0 {
			return one()
		}
		return b.outer
	}
	pair := l.gateUp.raw(mlx.Reshape(flat, N, 1, 1, H), idx) // [N, K, 1, 2*W]
	dt := x.Dtype()
	act := prepared(fusedKernel("kinfer_ink_act"), [10]int{N, K, W, flag(l.gateUp.outer != 0), int(dt)},
		func() ([]mlx.KernelOutput, []mlx.KernelTemplate, [3]int, [3]int) {
			return []mlx.KernelOutput{{Shape: []int{N, K, 1, W}, Dtype: dt}},
				[]mlx.KernelTemplate{{Name: "InT", Dtype: dt}, tplInt("NW", W), tplInt("NK", K), tplInt("HASO", flag(l.gateUp.outer != 0))},
				[3]int{N * K * W, 1, 1}, [3]int{256, 1, 1}
		}).Run(pair, idx, outer(l.gateUp), sig, mlx.Const(m.routeScale, mlx.Float32), mlx.Const(m.gateCap, mlx.Float32), mlx.Const(m.upCap, mlx.Float32))[0]
	down := l.down.raw(act, idx) // [N, K, 1, hidden]
	return prepared(fusedKernel("kinfer_ink_sum"), [10]int{B, L, H, K, flag(l.down.outer != 0), int(dt)},
		func() ([]mlx.KernelOutput, []mlx.KernelTemplate, [3]int, [3]int) {
			return []mlx.KernelOutput{{Shape: []int{B, L, H}, Dtype: dt}},
				[]mlx.KernelTemplate{{Name: "InT", Dtype: dt}, tplInt("NC", H), tplInt("NK", K), tplInt("HASO", flag(l.down.outer != 0))},
				[3]int{N * H, 1, 1}, [3]int{256, 1, 1}
		}).Run(down, idx, outer(l.down))[0]
}

// feedForwardSorted is the experts' part of feedForward for many tokens at
// once — a prompt being read: the tokens' choices are put in the order of the
// experts, so that an expert's weights are read once for all the tokens that
// chose it, where chosen token by token every choice reads its expert again.
func (m *Inkling2) feedForwardSorted(l *inkLayer, x, flat, idx, sig mlx.Array) mlx.Array {
	s := x.Shape()
	B, L, H := s[0], s[1], s[2]
	N, K, W := B*L, m.topK, m.width
	share := mlx.MulScalar(mlx.Divide(sig, mlx.Sum(sig, -1, true)), m.routeScale)
	choices := mlx.Reshape(idx, N*K)
	order := mlx.Argsort(choices, 0)
	rows := mlx.FloorDivide(order, mlx.ScalarAs(float32(K), order.Dtype()))
	sorted := mlx.Take(choices, order, 0)
	scaled := func(b inkBank, y mlx.Array) mlx.Array {
		y = mlx.Astype(y, mlx.Float32)
		if b.outer != 0 {
			y = mlx.Multiply(y, mlx.Reshape(mlx.Take(b.outer, sorted, 0), N*K, 1, 1))
		}
		return y
	}
	pair := mlx.Reshape(scaled(l.gateUp, l.gateUp.gather(mlx.Take(mlx.Reshape(flat, N, 1, H), rows, 0), sorted, true)), N*K, W, 2)
	gate := mlx.Slice(pair, []int{0, 0, 0}, []int{N * K, W, 1})
	up := mlx.Slice(pair, []int{0, 0, 1}, []int{N * K, W, 2})
	act := mlx.Multiply(mlx.Multiply(inkSoftcap(gate, m.gateCap), mlx.Sigmoid(gate)), inkSoftcap(up, m.upCap))
	act = mlx.Multiply(mlx.Reshape(act, N*K, 1, W), mlx.Reshape(mlx.Take(mlx.Reshape(share, N*K), order, 0), N*K, 1, 1))
	down := scaled(l.down, l.down.gather(mlx.Astype(act, x.Dtype()), sorted, true)) // [N*K, 1, hidden]
	back := mlx.Take(mlx.Reshape(down, N*K, H), mlx.Argsort(order, 0), 0)
	return mlx.Astype(mlx.Reshape(mlx.Sum(mlx.Reshape(back, N, K, H), 1, false), B, L, H), x.Dtype())
}

// gather is raw with idx said to be in ascending order or not.
func (b inkBank) gather(x, idx mlx.Array, sorted bool) mlx.Array {
	if b.scales == 0 {
		return mlx.GatherMM(x, mlx.Swapaxes(b.w, 1, 2), idx, sorted)
	}
	return mlx.GatherQMM(x, b.w, b.scales, 0, idx, b.group, b.bits, b.mode, sorted)
}

// raw is x times the experts idx names, as the weights are stored: without
// an expert's own scale, which the kernel after it applies.
func (b inkBank) raw(x, idx mlx.Array) mlx.Array {
	if b.scales == 0 {
		return mlx.GatherMM(x, mlx.Swapaxes(b.w, 1, 2), idx, false)
	}
	return mlx.GatherQMM(x, b.w, b.scales, 0, idx, b.group, b.bits, b.mode, false)
}

// mixGate is inkGate.apply's last step in one operation.
func mixGate(input, message, gf mlx.Array) mlx.Array {
	s := input.Shape()
	return fused("kinfer_ink_gate", input.Dtype(), size(s), []mlx.KernelOutput{{Shape: s, Dtype: mlx.Float32}},
		[]mlx.KernelTemplate{{Name: "MsgT", Dtype: message.Dtype()}, tplInt("NC", s[len(s)-1])}, input, message, gf)[0]
}

// The attention of one position, as decodeAttention has it for the Qwen3.5
// family (decode_attn.go): the first kernel splits the keys in view into
// chunks and gives each chunk of each head a threadgroup of NSG simdgroups,
// each keeping an online softmax over every NSG-th key; the second puts a
// head's chunks together. Here a key's score also takes the position's bias
// for how far back the key is, while its table reaches that far, and the
// second kernel gives the head's sink its share of the softmax before it
// divides, then multiplies by the sigmoid of the position's gate.
const inkAttnChunks = `
        uint lane = thread_position_in_threadgroup.x;
        uint g = thread_position_in_threadgroup.y;
        int s = int(threadgroup_position_in_grid.y);
        uint bh = threadgroup_position_in_grid.z;
        uint b = bh / NH;
        uint kvh = (bh % NH) / (NH / NHKV);
        constexpr int per = ND / 32;

        int lo = max(int(start[b]), first + s * chunk);
        int hi = min(end, first + (s + 1) * chunk);

        float qv[per];
        float acc[per];
        for (int i = 0; i < per; ++i) {
          qv[i] = static_cast<float>(q[bh * ND + lane * per + i]) * scale;
          acc[i] = 0.0f;
        }
        float m = -INFINITY;
        float l = 0.0f;
        uint64_t row = (uint64_t(b) * NHKV + kvh) * uint64_t(cap) * ND + lane * per;
        for (int j = lo + int(g); j < hi; j += NSG) {
          uint64_t at = row + uint64_t(j) * ND;
          float d = 0.0f;
          for (int i = 0; i < per; ++i) {
            d += qv[i] * static_cast<float>(k[at + i]);
          }
          int back = end - 1 - j;
          if (lane == 0 && back < NR) {
            // The position's bias for a key this far back: its relative
            // vector times the layer's row for the distance.
            for (int i = 0; i < NREL; ++i) {
              d += rel[bh * NREL + i] * pos[i * NR + back];
            }
          }
          d = simd_sum(d);
          float mn = max(m, d);
          float c = metal::exp(m - mn);
          float p = metal::exp(d - mn);
          l = l * c + p;
          for (int i = 0; i < per; ++i) {
            acc[i] = acc[i] * c + p * static_cast<float>(v[at + i]);
          }
          m = mn;
        }

        threadgroup float sm[NSG];
        threadgroup float sl[NSG];
        threadgroup float so[NSG * ND];
        if (lane == 0) {
          sm[g] = m;
          sl[g] = l;
        }
        for (int i = 0; i < per; ++i) {
          so[g * ND + lane * per + i] = acc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g != 0) {
          return;
        }
        float M = -INFINITY;
        for (int t = 0; t < NSG; ++t) {
          M = max(M, sm[t]);
        }
        float L = 0.0f;
        for (int i = 0; i < per; ++i) {
          acc[i] = 0.0f;
        }
        if (M > -INFINITY) {
          for (int t = 0; t < NSG; ++t) {
            float w = metal::exp(sm[t] - M);
            L += w * sl[t];
            for (int i = 0; i < per; ++i) {
              acc[i] += w * so[t * ND + lane * per + i];
            }
          }
        }
        uint o = bh * nsplit + s;
        if (lane == 0) {
          pm[o] = M;
          pl[o] = L;
        }
        for (int i = 0; i < per; ++i) {
          po[uint64_t(o) * ND + lane * per + i] = acc[i];
        }
`

const inkAttnCombine = `
        uint lane = thread_position_in_threadgroup.x;
        uint bh = thread_position_in_grid.z;
        uint b = bh / NH;
        uint h = bh % NH;
        constexpr int per = ND / 32;
        float sk = static_cast<float>(all[b * NW + OFFS + h]);
        float M = sk;
        for (int s = 0; s < nsplit; ++s) {
          M = max(M, pm[bh * nsplit + s]);
        }
        float L = metal::exp(sk - M);
        float acc[per];
        for (int i = 0; i < per; ++i) {
          acc[i] = 0.0f;
        }
        for (int s = 0; s < nsplit; ++s) {
          float w = metal::exp(pm[bh * nsplit + s] - M);
          L += w * pl[bh * nsplit + s];
          for (int i = 0; i < per; ++i) {
            acc[i] += w * po[(uint64_t(bh) * nsplit + s) * ND + lane * per + i];
          }
        }
        for (int i = 0; i < per; ++i) {
          uint at = h * ND + lane * per + i;
          float gt = static_cast<float>(all[b * NW + OFFG + at]);
          float yy = 1 / (1 + metal::precise::exp(metal::abs(gt)));
          float sig = (gt < 0) ? yy : 1 - yy;
          y[b * NH * ND + at] = static_cast<InT>(acc[i] / L * sig);
        }
`

// inkAttnPrep is everything between a position's projections and its
// attention, for one position a row: all [B, NW] holds, side by side, the
// query (NH·ND), the relative vector (NH·NREL), the sinks (NH), the gates
// (NH·ND), the key (NC = NHKV·ND) with its convolution's projected taps
// after it when DYN (NHKV·TAPS), and the value likewise. It gives the query
// and the relative vector scaled, shortened and sharpened (inkClip), the key
// and the value through their convolutions (inkConv.run) and then shortened
// or normed, and the two convolutions' histories — each rounding where the
// operations it stands for round.
const inkAttnPrep = `
        uint i = thread_position_in_grid.x % GRID;
        uint b = thread_position_in_grid.x / GRID;
        uint base = b * NW;
        constexpr uint OFFR = NH * ND;
        constexpr uint OFFK = NH * ND + NH * NREL + NH + NH * ND;
        constexpr uint KW = NC + (DYN ? NHKV * TAPS : 0);
        constexpr uint OFFV = OFFK + KW;
        if (i < NH * ND) {
          uint h = i / ND;
          float sq = 0.0f;
          for (uint d = 0; d < ND; ++d) {
            float v = static_cast<float>(all[base + h * ND + d]) * qs[h * ND + d];
            sq += v * v;
          }
          float f = metal::min(qlim / (metal::precise::sqrt(sq) + 1e-6f), 1.0f);
          float v = static_cast<float>(all[base + i]) * qs[i];
          qo[b * NH * ND + i] = static_cast<InT>(v * f * tau[b]);
        }
        if (i < NH * NREL) {
          uint h = i / NREL;
          float sq = 0.0f;
          for (uint d = 0; d < NREL; ++d) {
            float v = static_cast<float>(all[base + OFFR + h * NREL + d]) * rs[h * NREL + d];
            sq += v * v;
          }
          float f = metal::min(rlim / (metal::precise::sqrt(sq) + 1e-6f), 1.0f);
          float v = static_cast<float>(all[base + OFFR + i]) * rs[i];
          ro[b * NH * NREL + i] = static_cast<float>(static_cast<InT>(v * f * tau[b]));
        }
#define INK_CONV(off, hist, w, c, into) { \
          uint head_ = (c) / ND; \
          float sum_ = static_cast<float>(all[base + (off) + (c)]); \
          for (uint t = 0; t < TAPS; ++t) { \
            uint j_ = DYN ? (TAIL - t) : t; \
            float coef_ = w[t * NC + (c)]; \
            if (DYN) coef_ += static_cast<float>(all[base + (off) + NC + head_ * TAPS + t]); \
            float win_ = (j_ < TAIL) ? static_cast<float>(hist[(b * TAIL + j_) * NC + (c)]) : static_cast<float>(all[base + (off) + (c)]); \
            sum_ = sum_ + win_ * coef_; \
          } \
          into = static_cast<float>(static_cast<InT>(sum_ * kinv)); }
        if (i < NC) {
          uint head = i / ND;
          float ksq = 0.0f, vsq = 0.0f, mine_k = 0.0f, mine_v = 0.0f;
          for (uint d = 0; d < ND; ++d) {
            uint c = head * ND + d;
            float kc, vc;
            INK_CONV(OFFK, hk, wk, c, kc);
            INK_CONV(OFFV, hv, wv, c, vc);
            ksq += kc * kc;
            vsq += vc * vc;
            if (c == i) { mine_k = kc; mine_v = vc; }
          }
          float f = metal::min(qlim / (metal::precise::sqrt(ksq) + 1e-6f), 1.0f);
          ko[b * NC + i] = static_cast<InT>(mine_k * f);
          vo[b * NC + i] = static_cast<InT>(mine_v * metal::rsqrt(vsq / ND + eps));
        }
#undef INK_CONV
        if (i < TAIL * NC) {
          uint j = i / NC;
          uint c = i % NC;
          if (j + 1 < TAIL) {
            hko[(b * TAIL + j) * NC + c] = hk[(b * TAIL + j + 1) * NC + c];
            hvo[(b * TAIL + j) * NC + c] = hv[(b * TAIL + j + 1) * NC + c];
          } else {
            hko[(b * TAIL + j) * NC + c] = all[base + OFFK + c];
            hvo[(b * TAIL + j) * NC + c] = all[base + OFFV + c];
          }
        }
`

var (
	iaOnce              sync.Once
	iaChunks, iaCombine *mlx.Kernel
)

var iaPrep *mlx.Kernel

// inkAttend is one position's attention for each row: q [B, 1, H, D] as
// clipped, the rows' key and value buffers [B, Hkv, capacity, D] with row
// b's keys in view its columns start[b] to end-1 (start [B] int32, first the
// least of them), the position's relative vector rel [B, 1, H, d_rel] and
// the layer's positions [d_rel, range], both float32, whose product is the
// position's bias for each distance back, and its projections all [B, 1,
// width], which hold its sinks at offSink and its gates, before their
// sigmoid, at offGate. It returns the heads' outputs side by side, gated,
// [B, 1, H*D] in q's type. D must be a multiple of 32.
func inkAttend(q, keys, vals, rel, positions, all, start mlx.Array, offSink, offGate, first, end, H int) mlx.Array {
	inkAttnKernels()
	qs, ks := q.Shape(), keys.Shape()
	B, D := qs[0], qs[3]
	Hkv, capacity := ks[1], ks[2]
	ps := positions.Shape()
	nsplit := max(1, (end-first+decodeChunk-1)/decodeChunk)
	dt, width := q.Dtype(), all.Shape()[2]
	part := prepared(iaChunks, [10]int{B, H, D, Hkv, ps[0], ps[1], nsplit},
		func() ([]mlx.KernelOutput, []mlx.KernelTemplate, [3]int, [3]int) {
			return []mlx.KernelOutput{{Shape: []int{B * H * nsplit}, Dtype: mlx.Float32}, {Shape: []int{B * H * nsplit}, Dtype: mlx.Float32},
					{Shape: []int{B * H * nsplit, D}, Dtype: mlx.Float32}},
				[]mlx.KernelTemplate{tplInt("ND", D), tplInt("NH", H), tplInt("NHKV", Hkv), tplInt("NREL", ps[0]), tplInt("NR", ps[1]), tplInt("NSG", decodeGroups)},
				[3]int{32, decodeGroups * nsplit, B * H}, [3]int{32, decodeGroups, 1}
		}).Run(q, keys, vals, rel, positions, start, mlx.Int(first), mlx.Int(end), mlx.ConstInt(capacity), mlx.ConstInt(decodeChunk), mlx.ConstInt(nsplit),
		mlx.Const(float32(1/math.Sqrt(float64(D))), mlx.Float32))
	return prepared(iaCombine, [10]int{B, H, D, width, offSink, offGate, int(dt)},
		func() ([]mlx.KernelOutput, []mlx.KernelTemplate, [3]int, [3]int) {
			return []mlx.KernelOutput{{Shape: []int{B, 1, H * D}, Dtype: dt}},
				[]mlx.KernelTemplate{{Name: "InT", Dtype: dt}, tplInt("ND", D), tplInt("NH", H),
					tplInt("NW", width), tplInt("OFFS", offSink), tplInt("OFFG", offGate)},
				[3]int{32, 1, B * H}, [3]int{32, 1, 1}
		}).Run(part[0], part[1], part[2], all, mlx.ConstInt(nsplit))[0]
}

func inkAttnKernels() {
	iaOnce.Do(func() {
		iaChunks = mlx.NewKernel("kinfer_ink_attn_chunks",
			[]string{"q", "k", "v", "rel", "pos", "start", "first", "end", "cap", "chunk", "nsplit", "scale"},
			[]string{"pm", "pl", "po"}, inkAttnChunks)
		iaCombine = mlx.NewKernel("kinfer_ink_attn_combine",
			[]string{"pm", "pl", "po", "all", "nsplit"}, []string{"y"}, inkAttnCombine)
		iaPrep = mlx.NewKernel("kinfer_ink_attn_prep",
			[]string{"all", "hk", "hv", "wk", "wv", "qs", "rs", "tau", "qlim", "rlim", "kinv", "eps"},
			[]string{"qo", "ro", "ko", "vo", "hko", "hvo"}, inkAttnPrep)
	})
}

// prepFused is what a position's attention makes of its projections all [B,
// 1, width] before it attends: its query [B, 1, H, D] and relative vector
// [B, 1, H, d_rel] (float32), each row's sharpened by tau [B] float32, and
// its key and value [B, Hkv, 1, D], the two convolutions' histories moved on.
func (m *Inkling2) prepFused(a *inkAttention, all, tau mlx.Array, hk, hv *mlx.Array) (q, r, k, v mlx.Array) {
	inkAttnKernels()
	c := m.Config
	s := all.Shape()
	B, W := s[0], s[2]
	H, Hkv, D := a.heads, a.kvHeads, a.headDim
	NC, tail := Hkv*D, m.taps-1
	dyn := a.window == 0
	for _, h := range []*mlx.Array{hk, hv} {
		if *h == 0 {
			*h = mlx.Zeros(all.Dtype(), B, tail, NC)
		}
	}
	terms := m.taps + 1
	if dyn {
		terms += m.taps
	}
	grid := max(H*D, tail*NC)
	dt := all.Dtype()
	res := prepared(iaPrep, [10]int{B, W, H, D, m.relative, Hkv, m.taps, flag(dyn), int(dt)},
		func() ([]mlx.KernelOutput, []mlx.KernelTemplate, [3]int, [3]int) {
			return []mlx.KernelOutput{{Shape: []int{B, 1, H, D}, Dtype: dt}, {Shape: []int{B, 1, H, m.relative}, Dtype: mlx.Float32},
					{Shape: []int{B, Hkv, 1, D}, Dtype: dt}, {Shape: []int{B, Hkv, 1, D}, Dtype: dt},
					{Shape: []int{B, tail, NC}, Dtype: dt}, {Shape: []int{B, tail, NC}, Dtype: dt}},
				[]mlx.KernelTemplate{{Name: "InT", Dtype: dt}, tplInt("NW", W), tplInt("NH", H), tplInt("ND", D), tplInt("NREL", m.relative),
					tplInt("NHKV", Hkv), tplInt("NC", NC), tplInt("TAPS", m.taps), tplInt("TAIL", tail), tplInt("DYN", flag(dyn)), tplInt("GRID", grid)},
				[3]int{B * grid, 1, 1}, [3]int{256, 1, 1}
		}).Run(all, *hk, *hv, a.kConv.taps, a.vConv.taps, a.qScale, a.relScale, tau,
		mlx.Const(c.QKClip, mlx.Float32), mlx.Const(c.RelativeClip, mlx.Float32),
		mlx.Const(float32(1/math.Sqrt(float64(terms))), mlx.Float32), mlx.Const(m.normEps, mlx.Float32))
	*hk, *hv = res[4], res[5]
	return res[0], res[1], res[2], res[3]
}

// The kernels below each need sums over a token's whole hidden state — its
// size, a query's product with it — which one thread a number cannot give.
// A threadgroup of inkTG threads takes a token: each thread holds a run of
// the state's numbers, they leave their parts of a sum in the threadgroup's
// memory, and after a barrier every thread adds the parts up for itself.
const inkTG = 256

// inkSums is the opening every such kernel has: a thread's place, and SUM,
// which leaves this thread's part of sum k and gives the whole of it. All of
// a round's sums are left before any is read (INK_LEAVE, a barrier, INK_ALL),
// and read before the memory is used for the next round (another barrier).
const inkSums = `
        constexpr int TG = 256;
        constexpr int EPT = NH / TG;
        uint t = thread_position_in_threadgroup.x;
        uint n = thread_position_in_grid.z;
        threadgroup float part[TG * 16];
#define INK_LEAVE(k, v) part[(k) * TG + t] = (v)
#define INK_ALL(k, into) { float a_ = 0.0f; for (int u_ = 0; u_ < TG; ++u_) a_ += part[(k) * TG + u_]; into = a_; }
#define INK_BARRIER threadgroup_barrier(mem_flags::mem_threadgroup)
`

// inkMixSrc is inkMix and, with NORM, the norm after it: sources s0…s4, NS of
// them in use, each normed; queries [NH, NQ]; each query's softmax over the
// sources mixes them, and with NORM the mixture is normed again and scaled by
// nw [NQ, NH]. Mixture q goes to output q.
const inkMixSrc = inkSums + `
#define INK_SRC(s, i) ((s) == 0 ? static_cast<float>(s0[i]) : (s) == 1 ? static_cast<float>(s1[i]) : (s) == 2 ? static_cast<float>(s2[i]) : (s) == 3 ? static_cast<float>(s3[i]) : static_cast<float>(s4[i]))
        float xs[NS][EPT];
        for (int s = 0; s < NS; ++s) {
          float sq = 0.0f;
          for (int e = 0; e < EPT; ++e) {
            xs[s][e] = INK_SRC(s, n * NH + t * EPT + e);
            sq += xs[s][e] * xs[s][e];
          }
          INK_LEAVE(s, sq);
        }
        INK_BARRIER;
        float rn[NS];
        for (int s = 0; s < NS; ++s) {
          float all;
          INK_ALL(s, all);
          rn[s] = metal::rsqrt(all / NH + 1e-6f);
        }
        INK_BARRIER;
        for (int s = 0; s < NS; ++s) {
          for (int e = 0; e < EPT; ++e) xs[s][e] *= rn[s];
          for (int q = 0; q < NQ; ++q) {
            float dot = 0.0f;
            for (int e = 0; e < EPT; ++e) dot += xs[s][e] * queries[(t * EPT + e) * NQ + q];
            INK_LEAVE(q * NS + s, dot);
          }
        }
        INK_BARRIER;
        float w[NQ][NS];
        for (int q = 0; q < NQ; ++q) {
          float top = -INFINITY;
          for (int s = 0; s < NS; ++s) {
            INK_ALL(q * NS + s, w[q][s]);
            top = metal::max(top, w[q][s]);
          }
          float sum = 0.0f;
          for (int s = 0; s < NS; ++s) { w[q][s] = metal::precise::exp(w[q][s] - top); sum += w[q][s]; }
          for (int s = 0; s < NS; ++s) w[q][s] /= sum;
        }
        INK_BARRIER;
        float mixed[NQ][EPT];
        for (int q = 0; q < NQ; ++q) {
          float sq = 0.0f;
          for (int e = 0; e < EPT; ++e) {
            float v = 0.0f;
            for (int s = 0; s < NS; ++s) v += w[q][s] * xs[s][e];
            mixed[q][e] = v;
            sq += v * v;
          }
          INK_LEAVE(q, sq);
        }
        INK_BARRIER;
        for (int q = 0; q < NQ; ++q) {
          float scale = 1.0f;
          if (NORM) {
            float all;
            INK_ALL(q, all);
            scale = metal::rsqrt(all / NH + eps);
          }
          for (int e = 0; e < EPT; ++e) {
            uint c = t * EPT + e;
            float v = mixed[q][e] * scale;
            if (NORM) v = v * nw[q * NH + c];
            uint at = n * NH + c;
            if (q == 0) o0[at] = static_cast<OutT>(v);
            if (q == 1) o1[at] = static_cast<OutT>(v);
            if (q == 2) o2[at] = static_cast<OutT>(v);
            if (q == 3) o3[at] = static_cast<OutT>(v);
          }
        }
#undef INK_SRC
#undef INK_LEAVE
#undef INK_ALL
#undef INK_BARRIER
`

// inkGateInSrc is what a gate reads: its input and its message, each normed,
// side by side, [N, 2*NH] in OutT.
//
// (The four gates of a layer were also tried together: with their two small
// products as one batched product each, a step took 26 ms where this takes
// 21, MLX's product of three matrices at once being slower than three
// products; and with the products inside one kernel, 64 sums over the state
// each with its barrier, 50 ms.)
const inkGateInSrc = inkSums + `
        float xa[EPT];
        float xb[EPT];
        float sa = 0.0f, sb = 0.0f;
        for (int e = 0; e < EPT; ++e) {
          uint i = n * NH + t * EPT + e;
          xa[e] = static_cast<float>(input[i]);
          xb[e] = static_cast<float>(message[i]);
          sa += xa[e] * xa[e];
          sb += xb[e] * xb[e];
        }
        INK_LEAVE(0, sa);
        INK_LEAVE(1, sb);
        INK_BARRIER;
        float ta, tb;
        INK_ALL(0, ta);
        INK_ALL(1, tb);
        float ra = metal::rsqrt(ta / NH + 1e-6f), rb = metal::rsqrt(tb / NH + 1e-6f);
        for (int e = 0; e < EPT; ++e) {
          uint c = t * EPT + e;
          out[n * 2 * NH + c] = static_cast<OutT>(xa[e] * ra);
          out[n * 2 * NH + NH + c] = static_cast<OutT>(xb[e] * rb);
        }
#undef INK_LEAVE
#undef INK_ALL
#undef INK_BARRIER
`

// inkGateOutSrc is a gate's result, normed and scaled by nw [NH] for the
// layer that reads it: input·(1 − tanh(forget)) + message·tanh(gain), gf [N,
// 2*NH] holding gain then forget.
const inkGateOutSrc = inkSums + `
        float x[EPT];
        float sq = 0.0f;
        for (int e = 0; e < EPT; ++e) {
          uint c = t * EPT + e;
          float gain = metal::precise::tanh(gf[n * 2 * NH + c]);
          float forget = metal::precise::tanh(gf[n * 2 * NH + NH + c]);
          x[e] = static_cast<float>(input[n * NH + c]) * (1.0f - forget) + static_cast<float>(message[n * NH + c]) * gain;
          sq += x[e] * x[e];
        }
        INK_LEAVE(0, sq);
        INK_BARRIER;
        float all;
        INK_ALL(0, all);
        float scale = metal::rsqrt(all / NH + eps);
        for (int e = 0; e < EPT; ++e) {
          uint c = t * EPT + e;
          out[n * NH + c] = static_cast<OutT>(x[e] * scale * nw[c]);
        }
#undef INK_LEAVE
#undef INK_ALL
#undef INK_BARRIER
`

var (
	imOnce                     sync.Once
	imMix, imGateIn, imGateOut *mlx.Kernel
)

func inkGroupKernels() {
	imOnce.Do(func() {
		imMix = mlx.NewKernel("kinfer_ink_mix", []string{"s0", "s1", "s2", "s3", "s4", "queries", "nw", "eps"},
			[]string{"o0", "o1", "o2", "o3"}, inkMixSrc)
		imGateIn = mlx.NewKernel("kinfer_ink_gate_in", []string{"input", "message"}, []string{"out"}, inkGateInSrc)
		imGateOut = mlx.NewKernel("kinfer_ink_gate_out", []string{"input", "message", "gf", "nw", "eps"}, []string{"out"}, inkGateOutSrc)
	})
}

// inkGrouped says whether the threadgroup kernels take a state this wide.
func inkGrouped(hidden int) bool { return hidden%inkTG == 0 && hidden/inkTG <= 16 }

// inkMixFused is inkMix of sources [B, L, hidden] with queries [hidden, Q],
// one array a query, each [B, L, hidden]: float32 as inkMix gives them, or —
// with norms [Q, hidden] — normed and scaled by them in bfloat16, as inkRMS
// of each would be.
func inkMixFused(sources []mlx.Array, queries, norms mlx.Array, eps float32) []mlx.Array {
	inkGroupKernels()
	s := sources[0].Shape()
	B, L, H := s[0], s[1], s[2]
	Q := queries.Shape()[1]
	in := make([]mlx.Array, 5)
	for i := range in {
		in[i] = one()
		if i < len(sources) {
			in[i] = sources[i]
		}
	}
	outT, nw := mlx.Float32, one()
	if norms != 0 {
		outT, nw = mlx.Bfloat16, norms
	}
	S := len(sources)
	res := prepared(imMix, [10]int{B, L, H, S, Q, flag(norms != 0)},
		func() ([]mlx.KernelOutput, []mlx.KernelTemplate, [3]int, [3]int) {
			outs := make([]mlx.KernelOutput, 4)
			for q := range outs {
				outs[q] = mlx.KernelOutput{Shape: []int{1}, Dtype: outT}
				if q < Q {
					outs[q].Shape = []int{B, L, H}
				}
			}
			return outs, []mlx.KernelTemplate{{Name: "OutT", Dtype: outT}, tplInt("NH", H), tplInt("NS", S), tplInt("NQ", Q), tplInt("NORM", flag(norms != 0))},
				[3]int{inkTG, 1, B * L}, [3]int{inkTG, 1, 1}
		}).Run(in[0], in[1], in[2], in[3], in[4], queries, nw, mlx.Const(eps, mlx.Float32))
	return res[:Q]
}

// applyNormed is inkRMS(g.apply(input, message), norm, eps) in five
// operations: the two normed side by side, the two products, and the result
// normed.
func (g inkGate) applyNormed(input, message, norm mlx.Array, eps float32) mlx.Array {
	inkGroupKernels()
	s := input.Shape()
	B, L, H := s[0], s[1], s[2]
	setup := func(width int) func() ([]mlx.KernelOutput, []mlx.KernelTemplate, [3]int, [3]int) {
		return func() ([]mlx.KernelOutput, []mlx.KernelTemplate, [3]int, [3]int) {
			return []mlx.KernelOutput{{Shape: []int{B, L, width}, Dtype: mlx.Bfloat16}},
				[]mlx.KernelTemplate{{Name: "OutT", Dtype: mlx.Bfloat16}, tplInt("NH", H)},
				[3]int{inkTG, 1, B * L}, [3]int{inkTG, 1, 1}
		}
	}
	pair := prepared(imGateIn, [10]int{B, L, H}, setup(2*H)).Run(input, message)[0]
	gf := mlx.Matmul(mlx.Astype(mlx.Matmul(pair, g.down), mlx.Float32), g.both)
	return prepared(imGateOut, [10]int{B, L, H}, setup(H)).Run(input, message, gf, norm, mlx.Const(eps, mlx.Float32))[0]
}

// inkOnes is n ones, float32, kept: the sharpening of a layer that does not
// sharpen.
var inkOnesOf = map[int]mlx.Array{}

func inkOnes(n int) mlx.Array {
	if a, ok := inkOnesOf[n]; ok {
		return a
	}
	f := make([]float32, n)
	for i := range f {
		f[i] = 1
	}
	a := mlx.FromFloat32(f, n)
	mlx.Outlive(a)
	inkOnesOf[n] = a
	return a
}
