package llama

import (
	"testing"
	"unsafe"
)

// The structs libmtmd takes by value must match its own layout byte for byte:
// a field one place off reads as another, silently. The figures are what a C
// program compiled against b11998's mtmd.h printed.
func TestMtmdStructsMatchC(t *testing.T) {
	var p mtmdContextParams
	for _, c := range []struct {
		name      string
		got, want uintptr
	}{
		{"sizeof(mtmd_context_params)", unsafe.Sizeof(p), 96},
		{"device", unsafe.Offsetof(p.Device), 8},
		{"print_timings", unsafe.Offsetof(p.PrintTimings), 16},
		{"n_threads", unsafe.Offsetof(p.NThreads), 20},
		{"media_marker", unsafe.Offsetof(p.MediaMarker), 32},
		{"flash_attn_type", unsafe.Offsetof(p.FlashAttnType), 40},
		{"warmup", unsafe.Offsetof(p.Warmup), 44},
		{"image_max_tokens", unsafe.Offsetof(p.ImageMaxTokens), 52},
		{"cb_eval", unsafe.Offsetof(p.CbEval), 56},
		{"batch_max_tokens", unsafe.Offsetof(p.BatchMaxTokens), 72},
		{"progress_callback_user_data", unsafe.Offsetof(p.ProgressCallbackUserData), 88},
		{"sizeof(mtmd_input_text)", unsafe.Sizeof(mtmdInputText{}), 24},
		{"add_special", unsafe.Offsetof(mtmdInputText{}.addSpecial), 16},
		{"parse_special", unsafe.Offsetof(mtmdInputText{}.parseSpecial), 17},
		{"sizeof(mtmd_helper_init_opt)", unsafe.Sizeof(mtmdHelperInitOpt{}), 32},
		{"sizeof(mtmd_helper_bitmap_wrapper)", unsafe.Sizeof(mtmdBitmapWrapper{}), 16},
	} {
		if c.got != c.want {
			t.Errorf("%s = %d, C says %d", c.name, c.got, c.want)
		}
	}
}
