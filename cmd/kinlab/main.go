// Command kinlab is a bench for editing small models' weights and measuring
// what an edit does: kinfer's engines, models and store underneath, kept out
// of kinfer itself, which serves models and does not change them.
//
// An edited model is a directory of kinfer's store, links to its base's files
// and an edit.safetensors of the tensors that differ, so kinfer lists and
// serves it like any other.
package main

import (
	"fmt"
	"os"
)

const usage = `kinlab — edit a small model's weights, and measure what the edit does

  kinlab edit <op> <base> <name>  an edited model in kinfer's store: -from, -mix, -scale,
                                  -zero-neuron, -zero-head, -overlay; the base is left as it is
  kinlab eval [-tier t] <model> <set>
                                  run a set of problems many at once and grade the replies;
                                  tiers: smoke (seconds), ends (a minute), quick (minutes), full (an hour)
  kinlab eval -compare <a> <b>    two runs side by side: what each got right, and whether it is chance

Examples:
  kinlab edit -zero-neuron 15:0-63 Qwen3.5-0.8B zero64
  kinlab eval -tier smoke zero64 math500.json
  kinlab eval -tier quick zero64 math500.json   # its base is run too, once, and set beside it
`

func main() {
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "edit":
		err = cmdEdit(os.Args[2:])
	case "eval":
		err = cmdEval(os.Args[2:])
	case "-h", "--help", "help":
		fmt.Print(usage)
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "kinlab: %v\n", err)
		os.Exit(1)
	}
}
