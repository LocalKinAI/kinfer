package mlxmodel

import (
	"sync"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// The gated delta rule as one Metal kernel, from mlx-lm's gated_delta.py
// (MIT, Apple Inc.) — the scalar-gate kernel with its optional padding mask,
// which serves any Dk divisible by 32. One SIMD group of 32 threads owns one
// value row of one head's [Dv, Dk] state, each thread Dk/32 of its columns,
// and walks the T tokens in order: the recurrence stays in registers instead
// of making a dozen graph nodes per token, which is what makes prefill fast.
const gatedDeltaSource = `
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if (MASK) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          } else {
            y[dv_idx] = static_cast<InT>(0);
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
`

// stepsWrite has the kernel also write the state after each token, for a
// pass that checks drafts: the reply may keep only the first few of them,
// and the state after the last one kept is then at hand, where otherwise the
// pass would have to be read again up to it.
const stepsWrite = `
            auto st = states + ((uint64_t(b_idx) * T + t) * Hv + hv_idx) * Dv * Dk + uint64_t(dv_idx) * Dk;
            for (int i = 0; i < n_per_t; ++i) {
              st[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
            }
`

var (
	gdOnce             sync.Once
	gdKernel, gdMasked *mlx.Kernel
	gdSteps            *mlx.Kernel
)

func gatedDeltaKernels() (*mlx.Kernel, *mlx.Kernel) {
	gdOnce.Do(func() {
		in := []string{"q", "k", "v", "g", "beta", "state_in", "T"}
		out := []string{"y", "state_out"}
		gdKernel = mlx.NewKernel("kinfer_gated_delta", in, out, replaceMask(gatedDeltaSource, "true"))
		gdMasked = mlx.NewKernel("kinfer_gated_delta_mask", append(in, "mask"), out,
			replaceMask(gatedDeltaSource, "mask[b_idx * T + t]"))
		const after = "              y[dv_idx] = static_cast<InT>(out);\n            }\n"
		src := replaceMask(gatedDeltaSource, "true")
		i := indexOf(src, after)
		gdSteps = mlx.NewKernel("kinfer_gated_delta_steps", in, append(out, "states"),
			src[:i+len(after)]+stepsWrite+src[i+len(after):])
	})
	return gdKernel, gdMasked
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	panic("gated delta kernel source has changed: " + sub)
}

func replaceMask(src, cond string) string {
	const marker = "if (MASK)"
	for i := 0; i+len(marker) <= len(src); i++ {
		if src[i:i+len(marker)] == marker {
			return src[:i] + "if (" + cond + ")" + src[i+len(marker):]
		}
	}
	return src
}

// gatedDeltaFast is gatedDelta on the GPU kernel: q, k [B, T, Hk, Dk],
// v [B, T, Hv, Dv], g and beta [B, T, Hv], state [B, Hv, Dv, Dk] float32;
// mask [B, T] bool, or 0 when every position is real.
func gatedDeltaFast(q, k, v, g, beta, state, mask mlx.Array) (mlx.Array, mlx.Array) {
	s, vs := q.Shape(), v.Shape()
	B, T, Hk, Dk := s[0], s[1], s[2], s[3]
	Hv, Dv := vs[2], vs[3]
	plain, masked := gatedDeltaKernels()
	kern, inputs := plain, []mlx.Array{q, k, v, g, beta, state, mlx.Int(T)}
	if mask != 0 {
		kern, inputs = masked, append(inputs, mask)
	}
	res := kern.Run(inputs,
		[]mlx.KernelOutput{{Shape: []int{B, T, Hv, Dv}, Dtype: q.Dtype()}, {Shape: state.Shape(), Dtype: state.Dtype()}},
		[]mlx.KernelTemplate{{Name: "InT", Dtype: q.Dtype()}, {Name: "StT", Dtype: state.Dtype()},
			{Name: "Dk", Int: Dk, IsInt: true}, {Name: "Dv", Int: Dv, IsInt: true},
			{Name: "Hk", Int: Hk, IsInt: true}, {Name: "Hv", Int: Hv, IsInt: true}},
		[3]int{32, Dv, B * Hv}, [3]int{32, 4, 1})
	return res[0], res[1]
}

// gatedDeltaSteps is gatedDeltaFast for a pass that checks drafts, which
// also returns the state after each token, [B, T, Hv, Dv, Dk].
func gatedDeltaSteps(q, k, v, g, beta, state mlx.Array) (y, final, states mlx.Array) {
	s, vs := q.Shape(), v.Shape()
	B, T, Hk, Dk := s[0], s[1], s[2], s[3]
	Hv, Dv := vs[2], vs[3]
	gatedDeltaKernels()
	res := gdSteps.Run([]mlx.Array{q, k, v, g, beta, state, mlx.Int(T)},
		[]mlx.KernelOutput{{Shape: []int{B, T, Hv, Dv}, Dtype: q.Dtype()}, {Shape: state.Shape(), Dtype: state.Dtype()},
			{Shape: []int{B, T, Hv, Dv, Dk}, Dtype: state.Dtype()}},
		[]mlx.KernelTemplate{{Name: "InT", Dtype: q.Dtype()}, {Name: "StT", Dtype: state.Dtype()},
			{Name: "Dk", Int: Dk, IsInt: true}, {Name: "Dv", Int: Dv, IsInt: true},
			{Name: "Hk", Int: Hk, IsInt: true}, {Name: "Hv", Int: Hv, IsInt: true}},
		[3]int{32, Dv, B * Hv}, [3]int{32, 4, 1})
	return res[0], res[1], res[2]
}
