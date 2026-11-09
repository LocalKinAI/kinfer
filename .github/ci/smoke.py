#!/usr/bin/env python3
"""Drive a kinfer binary against a real model on whatever system CI gave us.

usage: smoke.py path/to/kinfer
"""
import http.client
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BIN = os.path.abspath(sys.argv[1])
SOURCE = "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL = "qwen2.5-0.5b-instruct-q4_k_m"
HOST, PORT = "127.0.0.1", 11599
LOGS = tempfile.mkdtemp()
failed = []


def check(name, ok, detail=""):
    print(("ok    " if ok else "FAIL  ") + name + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        failed.append(name)


def kinfer(*args, timeout=900):
    # To a file, not a pipe: a daemon this starts must not hold our read open.
    path = os.path.join(LOGS, f"cmd-{time.time_ns()}.log")
    with open(path, "wb") as out:
        rc = subprocess.run([BIN, *args], stdin=subprocess.DEVNULL, stdout=out,
                            stderr=subprocess.STDOUT, timeout=timeout).returncode
    with open(path, encoding="utf-8", errors="replace") as f:
        return rc, f.read()


def request(method, path, body=None, timeout=300):
    c = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    c.request(method, path, None if body is None else json.dumps(body),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read().decode("utf-8", "replace")
    c.close()
    return r.status, data


def chat(content, predict=40):
    status, data = request("POST", "/api/chat", {
        "model": MODEL, "stream": False,
        "options": {"num_predict": predict, "temperature": 0, "seed": 1},
        "messages": [{"role": "user", "content": content}],
    })
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {data[:300]}")
    return json.loads(data)


def sane(text):
    """Non-empty, and not a word said over and over, which is what a broken
    forward pass looks like: it still produces tokens, quickly."""
    return text.strip() != "" and re.search(r"(\w{2,}?)\s*(?:\1\s*){4,}", text) is None


def rate(d, count, duration):
    return d.get(count, 0) / max(d.get(duration, 1) / 1e9, 1e-9)


def story(tag):
    return f"Run {tag}. " + "The tide came in over the old stone pier and the gulls rose. " * 16 + \
        " Now write a long story about the sea."


rc, out = kinfer("version")
check("version", rc == 0, out.strip())

had_it = os.path.exists(os.path.join(os.path.expanduser("~"), ".kinfer", "models", MODEL + ".gguf"))
rc, out = kinfer("pull", SOURCE)
check("pull with checksum", rc == 0 and (had_it or "sha256 matches" in out), out.strip().splitlines()[-1] if out.strip() else "")

rc, out = kinfer("ls")
check("list", rc == 0 and MODEL in out)

rc, out = kinfer("fit")
check("fit", rc == 0, out.strip().splitlines()[0] if out.strip() else "")

rc, out = kinfer("can-run", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
print(out.strip()[:800], flush=True)

log_path = os.path.join(LOGS, "serve.log")
log = open(log_path, "wb")
server = subprocess.Popen([BIN, "serve", "-addr", f"{HOST}:{PORT}"],
                          stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
up = False
for _ in range(120):
    try:
        up = request("GET", "/api/version", timeout=2)[0] == 200
    except OSError:
        up = False
    if up:
        break
    time.sleep(0.5)
check("serve answers", up)

try:
    d = chat("What is the capital of France? One short sentence.")
    content = d["message"]["content"]
    check("/api/chat", "paris" in content.lower(), content.strip())

    rates = []
    first_prompt = 0
    for i in range(1, 9):
        d = chat(story(i), predict=128)
        rates.append(rate(d, "eval_count", "eval_duration"))
        if i == 1:
            first_prompt = rate(d, "prompt_eval_count", "prompt_eval_duration")
    print("decode tok/s over eight replies:", " ".join(f"{r:.0f}" for r in rates), flush=True)
    print(f"prompt tok/s: {first_prompt:.0f}", flush=True)
    early, late = statistics.mean(rates[:3]), statistics.mean(rates[-3:])
    check("decoding holds its speed", late >= early * 0.6, f"first three {early:.0f}, last three {late:.0f}")

    d = chat(story(8), predict=8)
    reread = rate(d, "prompt_eval_count", "prompt_eval_duration")
    check("a repeated prompt is reused", reread > first_prompt * 2, f"{reread:.0f} tok/s")

    status, data = request("POST", "/v1/chat/completions", {
        "model": MODEL, "stream": True, "max_tokens": 25, "temperature": 0,
        "messages": [{"role": "user", "content": "Name two colors."}]})
    pieces = ""
    for line in data.splitlines():
        if line.startswith("data: ") and "[DONE]" not in line:
            choices = json.loads(line[6:])["choices"]
            if choices and choices[0]["delta"].get("content"):
                pieces += choices[0]["delta"]["content"]
    check("/v1/chat/completions stream", status == 200 and sane(pieces), pieces.strip())

    status, data = request("POST", "/v1/messages", {
        "model": MODEL, "max_tokens": 20, "messages": [{"role": "user", "content": "Say hello."}]})
    check("/v1/messages", status == 200 and sane(json.loads(data)["content"][0]["text"]), data[:120])

    status, data = request("POST", "/v1/responses", {
        "model": MODEL, "max_output_tokens": 20, "input": "Say hello."})
    check("/v1/responses", status == 200 and sane(json.loads(data)["output"][0]["content"][0]["text"]), data[:120])

    replies = [None] * 3

    def one(i):
        try:
            replies[i] = chat(f"Write a poem about river number {i + 1}.", predict=24)["message"]["content"]
        except Exception as e:  # reported by the check below
            replies[i] = ""
            print("concurrent request failed:", e, flush=True)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("three at once", all(r and sane(r) for r in replies), repr([(r or "")[:30] for r in replies]))
except Exception as e:
    check("requests to the server", False, repr(e))

server.terminate()
try:
    server.wait(20)
    check("server stops", True)
except subprocess.TimeoutExpired:
    server.kill()
    check("server stops", False, "still running after 20 s")
log.close()

with open(log_path, encoding="utf-8", errors="replace") as f:
    lines = f.read().splitlines()
for line in lines:
    if "load_backend" in line:
        print(line)
if failed:
    print("--- end of the server's log")
    print("\n".join(lines[-40:]))

rc, out = kinfer("run", MODEL, "Reply with one word: ok", timeout=300)
check("run through a daemon", rc == 0 and "tok/s" in out, out.strip()[-160:])
rc, out = kinfer("ps")
check("ps shows the model", rc == 0 and MODEL in out)
rc, out = kinfer("stop")
check("stop", rc == 0 and "stopped" in out, out.strip())

print("\nFAILED: " + ", ".join(failed) if failed else "\nall passed", flush=True)
sys.exit(1 if failed else 0)
