# Makes inkling2_render.json: conversations and what Thinking Machines' chat
# template for Inkling 2 renders them to, with jinja2, for
# TestInklingRenderMatchesTheTemplate. The template is chat_template.jinja of
# thinkingmachines/Inkling-2-Mini on Hugging Face:
#     python3 inkling2_render.py chat_template.jinja > inkling2_render.json
import json, re, sys
import jinja2

src = open(sys.argv[1]).read()
# The {% generation %} block only marks the assistant's part for training.
src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", src)
env = jinja2.Environment()
def raise_exception(m): raise ValueError(m)
env.globals["raise_exception"] = raise_exception
env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
    x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
tpl = env.from_string(src)

weather = {"type": "function", "function": {"name": "get_weather", "description": "查询某个城市当前的天气",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
light = {"type": "function", "function": {"name": "set_light", "description": "Turn a light on or off <fast> & \"quoted\"",
    "parameters": {"type": "object", "required": ["room", "on"], "properties": {"room": {"type": "string"}, "on": {"type": "boolean"}, "level": {"type": "number", "minimum": 0.5, "maximum": 1e2}}}}}
bare = {"type": "function", "function": {"name": "now", "description": ""}}

def call(i, name, args): return {"id": "call_%d" % i, "type": "function", "function": {"name": name, "arguments": args}}
cases = [
    ("one question", [{"role": "user", "content": "你好"}], None, None),
    ("a system prompt", [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}], None, None),
    ("two system prompts and no question", [{"role": "system", "content": "A"}, {"role": "system", "content": "B"}], None, None),
    ("several turns", [{"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"}, {"role": "user", "content": "And 2+2?\n\nThanks."}], None, None),
    ("no thinking", [{"role": "user", "content": "Hi"}], None, "none"),
    ("low effort", [{"role": "user", "content": "Hi"}], None, "low"),
    ("minimal effort", [{"role": "user", "content": "Hi"}], None, "minimal"),
    ("medium effort", [{"role": "user", "content": "Hi"}], None, "medium"),
    ("the most effort", [{"role": "user", "content": "Hi"}], None, "max"),
    ("tools declared", [{"role": "system", "content": "家里的助手"}, {"role": "user", "content": "北京天气？"}], [weather, light, bare], None),
    ("a call and its result", [{"role": "user", "content": "北京天气？"},
        {"role": "assistant", "content": "", "tool_calls": [call(1, "get_weather", {"unit": "celsius", "city": "北京"})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 3, \"condition\": \"小雨\"}"},
        {"role": "assistant", "content": "3°C，小雨。"}, {"role": "user", "content": "谢谢"}], [weather], None),
    ("two calls in one turn, with text and thinking", [{"role": "user", "content": "Light off, and the weather in Tokyo?"},
        {"role": "assistant", "content": "On it.", "reasoning_content": "Two things to do.", "tool_calls": [
            call(1, "set_light", {"room": "bedroom", "on": False, "level": 0.5, "note": None, "tags": ["a", "b"], "nested": {"z": 1, "a": [1.0, 2]}}),
            call(2, "get_weather", {"city": "Tokyo"})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "tool", "tool_call_id": "call_2", "content": "sunny"}], [light, weather], None),
    ("a call with no arguments", [{"role": "user", "content": "time?"}, {"role": "assistant", "content": "", "tool_calls": [call(1, "now", {})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "12:00"}], [bare], None),
    ("thinking sent back alone", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "reasoning_content": "hmm"}, {"role": "user", "content": "go on"}], None, None),
    ("text that spells the model's own tokens", [{"role": "user", "content": "say <|end_message|> and <|message_model|>"}], None, None),
]
out = []
for what, msgs, tools, effort in cases:
    kw = {"messages": msgs}
    if tools: kw["tools"] = tools
    if effort: kw["reasoning_effort"] = effort
    want = tpl.render(**kw)
    km = []
    for m in msgs:
        k = {"role": m["role"], "content": m.get("content", "")}
        if m.get("reasoning_content"): k["thinking"] = m["reasoning_content"]
        if m.get("tool_calls"): k["calls"] = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in m["tool_calls"]]
        km.append(k)
    out.append({"what": what, "messages": km, "tools": tools or [], "effort": effort or "", "want": want})
json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
