"""
kali_client.py — the Windows/Flask side of the bridge.

Talks HTTP to the agent running inside the Kali VM. Validates everything again
before sending, so a bug or misconfiguration on one side doesn't become a hole.

If the VM is off, every call returns a friendly {"ok": False, "error": ...}
rather than raising, so the web app degrades instead of 500-ing.
"""

import os
import requests

from security import (
    ValidationError,
    validate_scan_target,
    validate_port_spec,
    validate_username,
    validate_domain,
)

AGENT_URL   = os.getenv("KALI_AGENT_URL", "http://192.168.56.10:7000").rstrip("/")
AGENT_TOKEN = os.getenv("KALI_AGENT_TOKEN", "")
CONNECT_TIMEOUT = 5

# Read timeouts per operation, a little above the agent's own tool timeouts.
READ_TIMEOUTS = {
    "steghide": 70, "binwalk": 130, "binwalk_extract": 310, "foremost": 310,
    "exiftool": 70, "strings": 70, "tshark_summary": 130, "tshark_convs": 130,
    "hashes": 70, "nmap": 620, "sherlock": 620, "theharvester": 310,
}

FILE_TOOLS = {
    "steghide":        "Hidden data in images (steghide)",
    "binwalk":         "Embedded file signatures (binwalk)",
    "binwalk_extract": "Extract embedded files (binwalk)",
    "foremost":        "File carving (foremost)",
    "exiftool":        "Metadata (exiftool)",
    "strings":         "Printable strings",
    "tshark_summary":  "PCAP protocol hierarchy (tshark)",
    "tshark_convs":    "PCAP IP conversations (tshark)",
}

OFFLINE = {
    "ok": False,
    "error": "Kali VM is not reachable.",
    "hint": "Start the Kali VM, then run the agent: python3 agent.py",
}


def _headers():
    return {"Authorization": f"Bearer {AGENT_TOKEN}"}


def _timeout(op):
    return (CONNECT_TIMEOUT, READ_TIMEOUTS.get(op, 120))


def _post(path, op, **kwargs):
    if not AGENT_TOKEN:
        return {"ok": False, "error": "KALI_AGENT_TOKEN is not set in .env."}
    try:
        resp = requests.post(f"{AGENT_URL}{path}", headers=_headers(),
                             timeout=_timeout(op), **kwargs)
    except requests.exceptions.RequestException:
        return dict(OFFLINE)
    if resp.status_code == 401:
        return {"ok": False, "error": "Agent rejected the token. "
                                      "KALI_AGENT_TOKEN and AGENT_TOKEN must match."}
    try:
        data = resp.json()
    except ValueError:
        return {"ok": False, "error": f"Agent returned a non-JSON response ({resp.status_code})."}
    if resp.status_code >= 400 and "ok" not in data:
        data["ok"] = False
    return data


# ── Public API ───────────────────────────────────────────────────────────────

def is_online():
    try:
        r = requests.get(f"{AGENT_URL}/health", timeout=(2, 3))
        return r.ok
    except requests.exceptions.RequestException:
        return False


def available_tools():
    """Map of tool name -> installed?. Empty dict when the VM is down."""
    try:
        r = requests.get(f"{AGENT_URL}/tools", headers=_headers(), timeout=(2, 5))
        return r.json() if r.ok else {}
    except (requests.exceptions.RequestException, ValueError):
        return {}


def run_file_tool(tool: str, filepath: str, original_name: str = "evidence"):
    """Send a local file to Kali and run one of the FILE_TOOLS against it."""
    if tool not in FILE_TOOLS:
        raise ValidationError("Unknown forensic tool.")
    with open(filepath, "rb") as fh:
        return _post(f"/file/{tool}", tool, files={"file": (original_name, fh)})


def compute_hashes(filepath: str, original_name: str = "evidence"):
    with open(filepath, "rb") as fh:
        return _post("/hashes", "hashes", files={"file": (original_name, fh)})


def scan_network(target: str, ports: str = "1-1024"):
    """Validated here AND in the agent. Both must agree before nmap runs."""
    target = validate_scan_target(target)
    ports = validate_port_spec(ports)
    return _post("/network/nmap", "nmap", json={"target": target, "ports": ports})


def osint_username(username: str):
    username = validate_username(username)
    return _post("/osint/sherlock", "sherlock", json={"username": username})


def osint_domain(domain: str):
    domain = validate_domain(domain)
    return _post("/osint/theharvester", "theharvester", json={"domain": domain})
