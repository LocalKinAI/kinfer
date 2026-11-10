package mlximage

import (
	"fmt"
	"math"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// The text encoder: Qwen3-4B read up to its second-to-last layer, in float32,
// as mflux's TextEncoder (all_hidden_states[-2]); the prompt in the chat
// template's user turn, padded to 512 tokens as its tokenizer pads it.
const (
	teHidden   = 2560
	teHeads    = 32
	teKV       = 8
	teHeadDim  = 128
	teLayers   = 35 // of 36: the last one's output is never read
	teEps      = 1e-6
	teHeadEps  = 1e-5 // q_norm and k_norm are nn.RMSNorm(head_dim), whose eps is MLX's default
	maxTokens  = 512
	padTokenID = 151643 // <|endoftext|>
)

type teLayer struct {
	inNorm, postNorm, qNorm, kNorm mlx.Array
	q, k, v, o, gate, up, down     linear
}

type textEncoder struct {
	emb     embedding
	layers  []teLayer
	invFreq mlx.Array
}

func loadTextEncoder(w *weights) *textEncoder {
	te := &textEncoder{emb: embedding{w.lin("embed_tokens")}, invFreq: w.get("rotary_emb.inv_freq")}
	for i := 0; i < teLayers; i++ {
		p := fmt.Sprintf("layers.%d.", i)
		te.layers = append(te.layers, teLayer{
			inNorm: w.get(p + "input_layernorm.weight"), postNorm: w.get(p + "post_attention_layernorm.weight"),
			qNorm: w.get(p + "self_attn.q_norm.weight"), kNorm: w.get(p + "self_attn.k_norm.weight"),
			q: w.lin(p + "self_attn.q_proj"), k: w.lin(p + "self_attn.k_proj"), v: w.lin(p + "self_attn.v_proj"),
			o: w.lin(p + "self_attn.o_proj"), gate: w.lin(p + "mlp.gate_proj"), up: w.lin(p + "mlp.up_proj"),
			down: w.lin(p + "mlp.down_proj"),
		})
	}
	return te
}

// promptIDs are the prompt's tokens as mflux's LanguageTokenizer makes them:
// the chat template's user turn and the assistant's opening (Qwen3's, with
// enable_thinking), at most 512.
func (m *Model) promptIDs(prompt string) []int32 {
	ids := m.tok.Encode("<|im_start|>user\n"+prompt+"<|im_end|>\n<|im_start|>assistant\n", true)
	if len(ids) > maxTokens {
		ids = ids[:maxTokens]
	}
	return ids
}

// rotateHalf is [-x2, x1], the halves of the last axis swapped.
func rotateHalf(x mlx.Array) mlx.Array {
	h := x.Shape()[3] / 2
	p := mlx.Split(x, 3, h)
	return mlx.Concatenate(3, mlx.Negative(p[1]), p[0])
}

// encode is the prompt's features [n, 2560] in bfloat16, one a token.
func (te *textEncoder) encode(ids []int32) mlx.Array {
	n := len(ids)
	padded := make([]int32, maxTokens)
	mask := make([]float32, maxTokens)
	for i := range padded {
		if i < n {
			padded[i] = ids[i]
		} else {
			padded[i], mask[i] = padTokenID, float32(math.Inf(-1))
		}
	}
	L := maxTokens
	h := mlx.Astype(te.emb.rows(mlx.FromInt32(padded, 1, L)), mlx.Float32)

	// RotaryEmbedding: the positions' angles, cos and sin in h's dtype.
	freqs := mlx.Multiply(mlx.Reshape(mlx.Arange(0, float64(L), 1, mlx.Float32), L, 1), mlx.Reshape(te.invFreq, 1, teHeadDim/2))
	emb := mlx.Concatenate(1, freqs, freqs)
	cos := mlx.Reshape(mlx.Astype(mlx.Cos(emb), h.Dtype()), 1, L, 1, teHeadDim)
	sin := mlx.Reshape(mlx.Astype(mlx.Sin(emb), h.Dtype()), 1, L, 1, teHeadDim)

	// The causal mask, and the padding's.
	causal := make([]float32, L*L)
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			if j > i {
				causal[i*L+j] = float32(math.Inf(-1))
			}
		}
	}
	am := mlx.Add(mlx.FromFloat32(causal, 1, 1, L, L), mlx.FromFloat32(mask, 1, 1, 1, L))

	scale := float32(math.Pow(teHeadDim, -0.5))
	for _, l := range te.layers {
		x := mlx.RMSNorm(h, l.inNorm, teEps)
		q := mlx.RMSNorm(mlx.Reshape(l.q.apply(x), 1, L, teHeads, teHeadDim), l.qNorm, teHeadEps)
		k := mlx.RMSNorm(mlx.Reshape(l.k.apply(x), 1, L, teKV, teHeadDim), l.kNorm, teHeadEps)
		v := mlx.Reshape(l.v.apply(x), 1, L, teKV, teHeadDim)
		q = mlx.Add(mlx.Multiply(q, cos), mlx.Multiply(rotateHalf(q), sin))
		k = mlx.Add(mlx.Multiply(k, cos), mlx.Multiply(rotateHalf(k), sin))
		k = mlx.Repeat(k, teHeads/teKV, 2)
		v = mlx.Repeat(v, teHeads/teKV, 2)
		o := mlx.SDPA(mlx.Transpose(q, 0, 2, 1, 3), mlx.Transpose(k, 0, 2, 1, 3), mlx.Transpose(v, 0, 2, 1, 3), scale, "", am)
		o = mlx.Reshape(mlx.Transpose(o, 0, 2, 1, 3), 1, L, teHeads*teHeadDim)
		h = mlx.Add(h, l.o.apply(o))
		x = mlx.RMSNorm(h, l.postNorm, teEps)
		h = mlx.Add(h, l.down.apply(mlx.Multiply(mlx.SiLU(l.gate.apply(x)), l.up.apply(x))))
	}
	h = mlx.Astype(h, mlx.Bfloat16)
	return mlx.Reshape(mlx.Slice(h, []int{0, 0, 0}, []int{1, n, teHidden}), n, teHidden)
}
