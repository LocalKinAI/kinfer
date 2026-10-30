package llama

// ─── which experts generation uses ──────────────────────────────────────────

var (
	moeCacheSaveProfile func(c Context, path *byte) bool
	moeCacheLoadProfile func(c Context, path *byte) bool
)

// bindProfile resolves the MoE cache profile functions. They are kinfer's own
// addition to llama.cpp (patches/llama.cpp), so a build from a llama.cpp
// release lacks them and kinfer runs without a profile.
func bindProfile(lib uintptr) {
	if register(&moeCacheSaveProfile, lib, "llama_moe_cache_save_profile") != nil ||
		register(&moeCacheLoadProfile, lib, "llama_moe_cache_load_profile") != nil {
		moeCacheSaveProfile, moeCacheLoadProfile = nil, nil
	}
}

// MoeProfiles reports whether this build can keep a profile of the experts a
// mixture of experts uses.
func MoeProfiles() bool { return moeCacheSaveProfile != nil }

// SaveMoeProfile merges the counts of the experts c's MoE cache saw selected
// since the last save into the file at path, halving what the file held. It is
// false when c has no MoE cache or the file cannot be written, and must not run
// while c decodes.
func SaveMoeProfile(c Context, path string) bool {
	if moeCacheSaveProfile == nil {
		return false
	}
	p := append([]byte(path), 0)
	return moeCacheSaveProfile(c, &p[0])
}

// LoadMoeProfile fills c's fresh MoE cache with the experts the file at path
// counts most. It is false when c has no MoE cache, or the file is missing or
// belongs to another model.
func LoadMoeProfile(c Context, path string) bool {
	if moeCacheLoadProfile == nil {
		return false
	}
	p := append([]byte(path), 0)
	return moeCacheLoadProfile(c, &p[0])
}
