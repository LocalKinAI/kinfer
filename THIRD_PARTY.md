# Third-party software in kinfer's Linux and Windows packages

The `kinfer` binary carries these libraries and unpacks them on first use.

| Library | Where it is used | License | Source |
|---|---|---|---|
| llama.cpp and ggml, build b11998 | Linux, Windows | MIT | <https://github.com/ggml-org/llama.cpp/releases/tag/b11998> |
| GNU OpenMP runtime, `libgomp.so.1`, 12.3.0-1ubuntu1~22.04.3 | Linux | GPL-3.0-or-later with the GCC Runtime Library Exception | <https://launchpad.net/ubuntu/+source/gcc-12/12.3.0-1ubuntu1~22.04.3> |
| LLVM OpenMP runtime, `libomp.dll`, as llama.cpp's release ships it | Windows | Apache-2.0 WITH LLVM-exception | <https://github.com/llvm/llvm-project> |

kinfer itself is under the license in `LICENSE`.
