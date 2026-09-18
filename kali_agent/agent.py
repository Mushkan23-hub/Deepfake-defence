#!/usr/bin/env python3
"""
agent.py — runs INSIDE the Kali VM. Not on Windows.

This replaces the "Flask sends a command over SSH" design. Instead of shipping
strings to a shell, the Windows app makes an HTTP call to a fixed endpoint here,
and this file decides which binary runs and with which arguments.

Why this is safer than SSH:
  - There is no shell. Every tool runs via subprocess.run([...], shell=False),
    so ';', '|', '$()' and backticks in user input are just bytes in a filename.
  - The set of runnable binaries is a hardcoded dict. A caller cannot name a
    binary, add a flag, or reach a tool that isn't listed.
  - Every run has a timeout and an output cap, so a hung or chatty tool cannot
    take the box down.

Listen address: bind to the VirtualBox host-only adapter only (default
192.168.56.10), never 0.0.0.0. Auth is a shared bearer token from AGENT_TOKEN.

Run:  python3 agent.py
"""

import os
import re
import shutil
import secrets
import subprocess
import tempfile
import hashlib
import json
from pathlib import Path

from flask import Flask, request, jsonify

# ── Config ───────────────────────────────────────────────────────────────────

BIND_HOST   = os.getenv("AGENT_HOST", "192.168.56.10")
BIND_PORT   = int(os.getenv("AGENT_PORT", "7000"))
AGENT_TOKEN = os.getenv("AGENT_TOKEN", "")
WORK_DIR    = Path(os.getenv("AGENT_WORK_DIR", "/tmp/dd-agent"))
MAX_UPLOAD  = 100 * 1024 * 1024
MAX_OUTPUT  = 512 * 1024          # truncate tool stdout beyond this

if not AGENT_TOKEN:
    raise SystemExit(
        "AGENT_TOKEN is not set.\n"
        "Generate one:  python3 -c \"import secrets;print(secrets.token_hex(32))\"\n"
        "Then:          export AGENT_TOKEN=<that value>\n"
        "The same value goes in your Windows .env as KALI_AGENT_TOKEN."
    )

WORK_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD


# ── Auth ─────────────────────────────────────────────────────────────────────

@app.before_request
def require_token():
    if request.path == "/health":
        return None
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return jsonify({"error": "Missing bearer token."}), 401
    if not secrets.compare_digest(header[7:], AGENT_TOKEN):
        return jsonify({"error": "Bad token."}), 401
    return None


# ── Tool registry ────────────────────────────────────────────────────────────
#
# build(...) returns a list of argv tokens. Note that no function here ever
# concatenates user input into a string — the untrusted value is always a
# single element in a list, which the kernel passes to execve() verbatim.

def _steghide(path, **kw):
    # Empty passphrase; steghide asks interactively without -p.
    return ["steghide", "info", "-p", "", str(path)]


def _binwalk(path, **kw):
    return ["binwalk", "--term", str(path)]


def _binwalk_extract(path, outdir, **kw):
    return ["binwalk", "--extract", "--directory", str(outdir), str(path)]


def _foremost(path, outdir, **kw):
    return ["foremost", "-i", str(path), "-o", str(outdir), "-T"]


def _exiftool(path, **kw):
    return ["exiftool", "-j", "-a", "-u", str(path)]


def _strings(path, **kw):
    return ["strings", "-n", "8", str(path)]


def _tshark_summary(path, **kw):
    return ["tshark", "-r", str(path), "-q", "-z", "io,phs"]


def _tshark_convs(path, **kw):
    return ["tshark", "-r", str(path), "-q", "-z", "conv,ip"]


def _nmap(target, ports, **kw):
    # -Pn skips host discovery; -sV grabs service banners; -T3 stays polite.
    return ["nmap", "-Pn", "-sV", "-T3", "-p", ports, "--", target]


def _sherlock(username, outdir, **kw):
    return ["sherlock", "--timeout", "10", "--print-found",
            "--folderoutput", str(outdir), "--", username]


def _theharvester(domain, **kw):
    return ["theHarvester", "-d", domain, "-b", "duckduckgo,crtsh,hackertarget", "-l", "100"]


# name -> (builder, timeout seconds, needs a file?, produces an output dir?)
TOOLS = {
    "steghide":        (_steghide,        60,   True,  False),
    "binwalk":         (_binwalk,         120,  True,  False),
    "binwalk_extract": (_binwalk_extract, 300,  True,  True),
    "foremost":        (_foremost,        300,  True,  True),
    "exiftool":        (_exiftool,        60,   True,  False),
    "strings":         (_strings,         60,   True,  False),
    "tshark_summary":  (_tshark_summary,  120,  True,  False),
    "tshark_convs":    (_tshark_convs,    120,  True,  False),
    "nmap":            (_nmap,            600,  False, False),
    "sherlock":        (_sherlock,        600,  False, True),
    "theharvester":    (_theharvester,    300,  False, False),
}

# Defence in depth: the Windows side validates these too. If one side is
# bypassed or misconfigured, the other still refuses.
ALLOWED_SCAN_HOSTS = {"scanme.nmap.org", "localhost", "127.0.0.1"}
ALLOWED_SCAN_PREFIXES = ("127.", "10.0.2.", "192.168.56.")
SAFE_TARGET = re.compile(r"^[A-Za-z0-9._-]{1,253}$")
SAFE_PORTS = re.compile(r"^\d{1,5}(-\d{1,5})?(,\d{1,5}(-\d{1,5})?)*$")


def _check_target(target):
    if not SAFE_TARGET.match(target or ""):
        raise ValueError("Invalid target.")
    if target in ALLOWED_SCAN_HOSTS:
        return target
    if target.startswith(ALLOWED_SCAN_PREFIXES):
        return target
    raise ValueError(f"{target} is not on the agent's scan allowlist.")


# ── Runner ───────────────────────────────────────────────────────────────────

def run_tool(argv, timeout):
    """Execute argv with no shell. Returns a structured result, never raises."""
    binary = shutil.which(argv[0])
    if not binary:
        return {"ok": False, "error": f"{argv[0]} is not installed in this VM.",
                "hint": f"sudo apt install -y {argv[0]}"}
    try:
        proc = subprocess.run(
            [binary] + argv[1:],
            shell=False,                 # <- the important bit
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            cwd=str(WORK_DIR),
            env={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                 "HOME": str(WORK_DIR), "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"{argv[0]} timed out after {timeout}s."}
    except Exception as exc:
        return {"ok": False, "error": f"{argv[0]} failed to start: {exc}"}

    out = proc.stdout or ""
    err = proc.stderr or ""
    truncated = len(out) > MAX_OUTPUT
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": out[:MAX_OUTPUT],
        "stderr": err[:8192],
        "truncated": truncated,
        "command": " ".join(argv),   # for the case log / court trail, display only
    }


def save_upload(file_storage):
    """Store an upload under a random name inside WORK_DIR. Never trust the name."""
    ext = ""
    if "." in (file_storage.filename or ""):
        raw = file_storage.filename.rsplit(".", 1)[1].lower()
        if re.fullmatch(r"[a-z0-9]{1,8}", raw):
            ext = "." + raw
    path = WORK_DIR / f"{secrets.token_hex(16)}{ext}"
    file_storage.save(str(path))
    return path


def hash_file(path):
    md5, sha1, sha256 = hashlib.md5(), hashlib.sha1(), hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            md5.update(chunk); sha1.update(chunk); sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha1": sha1.hexdigest(),
            "sha256": sha256.hexdigest(), "size_bytes": os.path.getsize(path)}


def list_tree(root, limit=200):
    out = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            out.append({"name": str(p.relative_to(root)), "size": p.stat().st_size})
        if len(out) >= limit:
            break
    return out


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "up", "service": "deepfake-defence kali agent"})


@app.get("/tools")
def tools():
    """Which tools are actually installed. The UI greys out the missing ones."""
    return jsonify({name: bool(shutil.which(_probe_binary(name)))
                    for name in TOOLS})


def _probe_binary(name):
    return {"binwalk_extract": "binwalk",
            "tshark_summary": "tshark",
            "tshark_convs": "tshark"}.get(name, name)


@app.post("/file/<tool>")
def file_tool(tool):
    """Run a file-based tool against an uploaded file."""
    if tool not in TOOLS:
        return jsonify({"error": "Unknown tool."}), 404
    builder, timeout, needs_file, makes_dir = TOOLS[tool]
    if not needs_file:
        return jsonify({"error": f"{tool} does not take a file."}), 400
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    path = save_upload(request.files["file"])
    outdir = None
    try:
        digests = hash_file(path)
        if makes_dir:
            outdir = Path(tempfile.mkdtemp(dir=WORK_DIR))
            result = run_tool(builder(path=path, outdir=outdir), timeout)
            result["artifacts"] = list_tree(outdir)
        else:
            result = run_tool(builder(path=path), timeout)
        result["hashes"] = digests
        result["tool"] = tool
        return jsonify(result)
    finally:
        path.unlink(missing_ok=True)
        if outdir:
            shutil.rmtree(outdir, ignore_errors=True)


@app.post("/hashes")
def hashes():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    path = save_upload(request.files["file"])
    try:
        return jsonify({"ok": True, "tool": "hashes", "hashes": hash_file(path)})
    finally:
        path.unlink(missing_ok=True)


@app.post("/network/nmap")
def nmap_scan():
    body = request.get_json(silent=True) or {}
    try:
        target = _check_target((body.get("target") or "").strip().lower())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    ports = (body.get("ports") or "1-1024").strip()
    if not SAFE_PORTS.match(ports):
        return jsonify({"error": "Invalid port specification."}), 400

    builder, timeout, _, _ = TOOLS["nmap"]
    result = run_tool(builder(target=target, ports=ports), timeout)
    result["tool"] = "nmap"
    result["target"] = target
    return jsonify(result)


@app.post("/osint/<tool>")
def osint(tool):
    if tool not in ("sherlock", "theharvester"):
        return jsonify({"error": "Unknown tool."}), 404
    body = request.get_json(silent=True) or {}
    builder, timeout, _, makes_dir = TOOLS[tool]

    outdir = None
    try:
        if tool == "sherlock":
            username = (body.get("username") or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9._-]{2,39}", username):
                return jsonify({"error": "Invalid username."}), 400
            outdir = Path(tempfile.mkdtemp(dir=WORK_DIR))
            result = run_tool(builder(username=username, outdir=outdir), timeout)
            result["subject"] = username
        else:
            domain = (body.get("domain") or "").strip().lower()
            if not re.fullmatch(r"[A-Za-z0-9.-]{4,253}", domain) or "." not in domain:
                return jsonify({"error": "Invalid domain."}), 400
            result = run_tool(builder(domain=domain), timeout)
            result["subject"] = domain
        result["tool"] = tool
        return jsonify(result)
    finally:
        if outdir:
            shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    print(f"Kali agent listening on http://{BIND_HOST}:{BIND_PORT}")
    print("Bind address should be your host-only adapter, never 0.0.0.0.")
    app.run(host=BIND_HOST, port=BIND_PORT, debug=False)
