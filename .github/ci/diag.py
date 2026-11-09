#!/usr/bin/env python3
"""On one machine, ask llama.cpp's own server and kinfer the same questions, so
that a wrong reply can be pinned on a layer.

usage: diag.py kinfer kinfer-no-avx512 upstream-dir
"""
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KINFER, NO512, UPSTREAM = [os.path.abspath(a) for a in sys.argv[1:4]]
MODEL = "qwen2.5-0.5b-instruct-q4_k_m"
FILE = os.path.join(os.path.expanduser("~"), ".kinfer", "models", MODEL + ".gguf")
PORT = 11601
LOGS = tempfile.mkdtemp()
QUESTIONS = [
    "What is the capital of France? One short sentence.",
    "Name two colors.",
    "Write a poem about a river.",
]


def get(path):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3)
    c.request("GET", path)
    r = c.getresponse()
    r.read()
    c.close()
    return r.status


def ask(q):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
    c.request("POST", "/v1/chat/completions", json.dumps({
        "model": MODEL, "max_tokens": 40, "temperature": 0, "seed": 1,
        "messages": [{"role": "user", "content": q}]}), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read().decode("utf-8", "replace")
    c.close()
    if r.status != 200:
        return f"HTTP {r.status}: {data[:200]}"
    return json.loads(data)["choices"][0]["message"]["content"]


def run(label, cmd, ready, env=None, fresh_libs=False):
    print(f"\n=== {label}", flush=True)
    if fresh_libs:
        shutil.rmtree(os.path.join(os.environ["LOCALAPPDATA"], "kinfer", "lib"), ignore_errors=True)
    path = os.path.join(LOGS, f"{time.time_ns()}.log")
    log = open(path, "wb")
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                         env={**os.environ, **(env or {})})
    up = False
    for _ in range(240):
        try:
            up = get(ready) == 200
        except OSError:
            up = False
        if up or p.poll() is not None:
            break
        time.sleep(0.5)
    if up:
        for q in QUESTIONS:
            try:
                print("  ", repr(ask(q))[:150], flush=True)
            except Exception as e:
                print("   request failed:", repr(e), flush=True)
    else:
        print("   never came up; exit", p.poll(), flush=True)
    p.terminate()
    try:
        p.wait(20)
    except subprocess.TimeoutExpired:
        p.kill()
    log.close()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if any(k in line for k in ("loaded CPU backend", "system_info", "n_threads", "CPU :")):
                print("  ", line.strip()[:210])
    sys.stdout.flush()


subprocess.run(["powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | Format-List | Out-String).Trim()"])
subprocess.run([KINFER, "pull", "Qwen/Qwen2.5-0.5B-Instruct-GGUF:" + MODEL + ".gguf"], stdout=subprocess.DEVNULL)

server = os.path.join(UPSTREAM, "llama-server.exe")
base = [server, "-m", FILE, "--port", str(PORT), "--host", "127.0.0.1", "-np", "1"]
run("upstream llama-server, its defaults", base, "/health")
run("upstream, --no-repack", base + ["--no-repack"], "/health")
run("upstream, -t 2 -tb 2", base + ["-t", "2", "-tb", "2"], "/health")

kinfer = [KINFER, "serve", "-addr", f"127.0.0.1:{PORT}"]
run("kinfer, its defaults", kinfer, "/api/version", fresh_libs=True)
run("kinfer, one thread", kinfer, "/api/version", env={"GOMAXPROCS": "1"})
run("kinfer, -slots 1 -prefix 0", kinfer + ["-slots", "1", "-prefix", "0"], "/api/version")
run("kinfer without the AVX-512 libraries", [NO512, "serve", "-addr", f"127.0.0.1:{PORT}"],
    "/api/version", fresh_libs=True)
