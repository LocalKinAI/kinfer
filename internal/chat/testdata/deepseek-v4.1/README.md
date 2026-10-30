# DeepSeek V4.1 encoder fixtures

Copied unchanged from `encoding/tests/` in
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(MIT), where DeepSeek's own encoder, `encoding/encoding.py`, is tested against
them. Each `test_input_N.json` is a conversation and `test_output_N.txt` the
exact prompt that encoder writes for it, rendered in chat mode unless the input
says `"thinking_mode": "thinking"`.

`deepseek41_test.go` renders the same conversations with kinfer and compares
byte for byte. Case 3 is left out: it attaches its tools to a system message in
the middle of the conversation, which kinfer's API cannot express — tools there
belong to the request. Case 4 is compared up to the `<｜action｜>` task token
its last user turn carries, which kinfer has no field for either.
