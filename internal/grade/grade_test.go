package grade

import "testing"

func TestBoxed(t *testing.T) {
	for _, c := range []struct{ text, want string }{
		{`so \boxed{\frac{1}{2}}.`, `\frac{1}{2}`},
		{`first \boxed{3}, then \boxed{4}`, `4`},
		{`\boxed {x^{2}}`, `x^{2}`},
	} {
		if got, ok := Boxed(c.text); !ok || got != c.want {
			t.Errorf("Boxed(%q) = %q, %v; want %q", c.text, got, ok, c.want)
		}
	}
	for _, text := range []string{"no box", `\boxed{unclosed`} {
		if got, ok := Boxed(text); ok {
			t.Errorf("Boxed(%q) = %q, want nothing", text, got)
		}
	}
}

func TestSame(t *testing.T) {
	for _, c := range []struct{ got, want string }{
		{"0.5", `\frac{1}{2}`},
		{`\dfrac{3}{4}`, `\frac34`},
		{`\frac{3}{4}`, "3/4"},
		{"0.75", `\frac{3}{4}`},
		{`10\%`, "10"},
		{"x = 5", "5"},
		{"1,000,000", "1000000"},
		{`\left( 3, \frac{\pi}{2} \right)`, `(3,\frac{\pi}{2})`},
		{`\text{(C)}`, "(C)"},
		{`90^\circ`, "90"},
		{`\sqrt2`, `\sqrt{2}`},
		{`\$18.90`, "18.90"},
		{".5", `\frac{1}{2}`},
		{"-2", "-2.0"},
	} {
		if !Same(c.got, c.want) {
			t.Errorf("Same(%q, %q) = false; stripped %q and %q", c.got, c.want, Strip(c.got), Strip(c.want))
		}
	}
	for _, c := range []struct{ got, want string }{
		{"3", "4"},
		{`\frac{1}{3}`, "0.33"},
		{"(1,2)", "(2,1)"},
		{"", "0"},
		{"xy = 3", "3"},
	} {
		if Same(c.got, c.want) {
			t.Errorf("Same(%q, %q) = true", c.got, c.want)
		}
	}
}
