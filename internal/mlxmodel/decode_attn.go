package mlxmodel

import (
	"sync"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// A batch's rows share one KV buffer as wide as its longest row, and MLX's
// attention reads all of it for every row, masked or not: one row thinking at
// 30,000 tokens makes every other row of a 64-row step read 30,000 columns,
// most of them padding. These two kernels read each row's own columns only.
//
// The first splits a row's columns into chunks and gives each chunk of each
// head a threadgroup of NSG simdgroups, each reading every NSG-th key, so
// that many keys are in flight at once. A simdgroup keeps an online softmax
// in float32 — the running maximum, the sum of exponentials and the weighted
// sum of values — with each lane holding D/32 of the head's dimensions, and
// the threadgroup's first simdgroup puts the others' together. Chunks outside
// the row write nothing to count. The second kernel combines a head's
// chunks, rescaling each by its maximum.
const decodeAttnChunks = `
        uint lane = thread_position_in_threadgroup.x;
        uint g = thread_position_in_threadgroup.y;
        int s = int(threadgroup_position_in_grid.y);
        uint bh = threadgroup_position_in_grid.z;
        uint b = bh / H;
        uint kvh = (bh % H) / (H / Hkv);
        constexpr int per = D / 32;

        int lo = max(start[b], s * chunk);
        int hi = min(end, (s + 1) * chunk);

        float qv[per];
        float acc[per];
        for (int i = 0; i < per; ++i) {
          qv[i] = static_cast<float>(q[bh * D + lane * per + i]) * scale;
          acc[i] = 0.0f;
        }
        float m = -INFINITY;
        float l = 0.0f;
        uint64_t row = (uint64_t(b) * Hkv + kvh) * uint64_t(cap) * D + lane * per;
        for (int j = lo + int(g); j < hi; j += NSG) {
          uint64_t at = row + uint64_t(j) * D;
          float d = 0.0f;
          for (int i = 0; i < per; ++i) {
            d += qv[i] * static_cast<float>(k[at + i]);
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

        // The threadgroup's simdgroups read every NSG-th key; the first of
        // them puts their parts together.
        threadgroup float sm[NSG];
        threadgroup float sl[NSG];
        threadgroup float so[NSG * D];
        if (lane == 0) {
          sm[g] = m;
          sl[g] = l;
        }
        for (int i = 0; i < per; ++i) {
          so[g * D + lane * per + i] = acc[i];
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
              acc[i] += w * so[t * D + lane * per + i];
            }
          }
        }
        uint o = bh * nsplit + s;
        if (lane == 0) {
          pm[o] = M;
          pl[o] = L;
        }
        for (int i = 0; i < per; ++i) {
          po[uint64_t(o) * D + lane * per + i] = acc[i];
        }
`

const decodeAttnCombine = `
        uint lane = thread_position_in_threadgroup.x;
        uint bh = thread_position_in_grid.z;
        constexpr int per = D / 32;
        float M = -INFINITY;
        for (int s = 0; s < nsplit; ++s) {
          M = max(M, pm[bh * nsplit + s]);
        }
        float L = 0.0f;
        float acc[per];
        for (int i = 0; i < per; ++i) {
          acc[i] = 0.0f;
        }
        for (int s = 0; s < nsplit; ++s) {
          float w = metal::exp(pm[bh * nsplit + s] - M);
          L += w * pl[bh * nsplit + s];
          for (int i = 0; i < per; ++i) {
            acc[i] += w * po[(uint64_t(bh) * nsplit + s) * D + lane * per + i];
          }
        }
        for (int i = 0; i < per; ++i) {
          y[bh * D + lane * per + i] = static_cast<T>(acc[i] / L);
        }
`

// raggedDecode chooses decodeAttention over MLX's masked attention; a test
// turns it off to time the two.
var raggedDecode = true

// decodeChunk is how many of a row's columns one threadgroup reads, and
// decodeGroups how many simdgroups it reads them with, a key each at a time.
const (
	decodeChunk  = 1024
	decodeGroups = 16
)

var (
	daOnce              sync.Once
	daChunks, daCombine *mlx.Kernel
)

// decodeAttention is one decode step's attention for a batch: q [B, H, 1, D],
// the rows' key and value buffers [B, Hkv, capacity, D], and for each row the
// first of its columns, start [B] int32; every row's last column is end-1.
// It returns [B, H*D], each row's heads side by side. D must be a multiple
// of 32.
func decodeAttention(q, keys, vals, start mlx.Array, end int, scale float32) mlx.Array {
	daOnce.Do(func() {
		daChunks = mlx.NewKernel("kinfer_decode_attn_chunks",
			[]string{"q", "k", "v", "start", "end", "cap", "chunk", "nsplit", "scale"},
			[]string{"pm", "pl", "po"}, decodeAttnChunks)
		daCombine = mlx.NewKernel("kinfer_decode_attn_combine",
			[]string{"pm", "pl", "po", "nsplit"}, []string{"y"}, decodeAttnCombine)
	})
	qs, ks := q.Shape(), keys.Shape()
	B, H, D := qs[0], qs[1], qs[3]
	Hkv, capacity := ks[1], ks[2]
	nsplit := (end + decodeChunk - 1) / decodeChunk
	tpl := []mlx.KernelTemplate{{Name: "T", Dtype: q.Dtype()}, {Name: "D", Int: D, IsInt: true},
		{Name: "H", Int: H, IsInt: true}, {Name: "Hkv", Int: Hkv, IsInt: true}, {Name: "NSG", Int: decodeGroups, IsInt: true}}
	part := daChunks.Run(
		[]mlx.Array{q, keys, vals, start, mlx.Int(end), mlx.Int(capacity), mlx.Int(decodeChunk), mlx.Int(nsplit), mlx.Scalar(scale)},
		[]mlx.KernelOutput{{Shape: []int{B * H * nsplit}, Dtype: mlx.Float32}, {Shape: []int{B * H * nsplit}, Dtype: mlx.Float32},
			{Shape: []int{B * H * nsplit, D}, Dtype: mlx.Float32}},
		tpl, [3]int{32, decodeGroups * nsplit, B * H}, [3]int{32, decodeGroups, 1})
	out := daCombine.Run(append(part, mlx.Int(nsplit)),
		[]mlx.KernelOutput{{Shape: []int{B, H * D}, Dtype: q.Dtype()}},
		[]mlx.KernelTemplate{{Name: "T", Dtype: q.Dtype()}, {Name: "D", Int: D, IsInt: true}},
		[3]int{32, 1, B * H}, [3]int{32, 1, 1})
	return out[0]
}
