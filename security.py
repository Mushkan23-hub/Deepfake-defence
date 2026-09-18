"""
security.py — input validation and authorisation gates.

Everything that crosses the boundary from a browser into a Kali tool passes
through this module first. Nothing here builds shell strings; the job is to
reject anything that isn't obviously safe, and let kali_client.py pass clean
values as argument lists.

Rule of thumb used throughout: allowlist, never denylist.
"""

import os
import re
import ipaddress

# ── Upload rules ─────────────────────────────────────────────────────────────

ALLOWED_IMAGE = {"png", "jpg", "jpeg", "webp", "bmp"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_AUDIO = {"wav", "mp3", "flac", "ogg", "m4a"}
ALLOWED_PCAP  = {"pcap", "pcapng"}

# Files we are willing to hand to a Kali carving/stego tool.
ALLOWED_FORENSIC = ALLOWED_IMAGE | ALLOWED_VIDEO | ALLOWED_AUDIO | ALLOWED_PCAP | {
    "zip", "pdf", "bin", "img", "dd", "raw", "doc", "docx"
}

MAX_UPLOAD_MB = 100

# Uploaded files are always renamed to <32 hex chars>.<ext>. Anything that does
# not match this shape never came from our own save_upload(), so we refuse it.
_STORED_NAME = re.compile(r"^[0-9a-f]{32}\.[A-Za-z0-9]{1,8}$")


class ValidationError(ValueError):
    """Raised when user input fails a check. Message is safe to show the user."""


def extension_of(filename: str) -> str:
    if "." not in filename:
        raise ValidationError("File has no extension.")
    return filename.rsplit(".", 1)[1].lower()


def check_extension(filename: str, allowed: set) -> str:
    ext = extension_of(filename)
    if ext not in allowed:
        raise ValidationError(
            f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(allowed))}"
        )
    return ext


def safe_stored_path(upload_dir: str, stored_name: str) -> str:
    """
    Resolve a stored filename to an absolute path, guaranteeing it stays inside
    upload_dir. Blocks '../', absolute paths, symlink escapes and null bytes.
    """
    if "\x00" in stored_name or not _STORED_NAME.match(stored_name):
        raise ValidationError("Invalid file reference.")

    root = os.path.realpath(upload_dir)
    path = os.path.realpath(os.path.join(root, stored_name))

    if os.path.commonpath([root, path]) != root:
        raise ValidationError("Invalid file reference.")
    if not os.path.isfile(path):
        raise ValidationError("File not found. It may have been cleaned up.")
    return path


# ── Network target rules ─────────────────────────────────────────────────────
#
# An nmap scan is traffic sent to someone else's machine. Unauthorised port
# scanning is an offence under the IT Act, 2000 (s.43 / s.66) in India and under
# equivalent law elsewhere. This app therefore refuses to scan anything that is
# not (a) an official public test host, (b) loopback, or (c) a private lab range
# that you control.
#
# Widening this list is a deliberate act. If you add a target, record written
# authorisation for it in the case file first.

ALLOWED_SCAN_HOSTS = {
    "scanme.nmap.org",      # Nmap project's official scan-permitted host
    "localhost",
    "127.0.0.1",
}

# Private ranges: your own VirtualBox host-only / NAT networks and home LAN.
ALLOWED_SCAN_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.2.0/24"),      # VirtualBox NAT default
    ipaddress.ip_network("192.168.56.0/24"),  # VirtualBox host-only default
]

_HOSTNAME = re.compile(
    r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)"
    r"(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*\.?$"
)


def validate_scan_target(target: str) -> str:
    """
    Return the target if it is on the allowlist, else raise. Accepts a bare IP
    or a hostname. No CIDR ranges, no comma lists, no nmap option syntax.
    """
    target = (target or "").strip().lower().rstrip(".")
    if not target or len(target) > 253:
        raise ValidationError("Enter a target host.")

    # Reject anything that looks like it is trying to smuggle flags or globs.
    if any(c in target for c in " \t\n\r;|&$`()<>*?![]{}'\"\\,/"):
        raise ValidationError("Target contains characters that are not allowed.")

    try:
        ip = ipaddress.ip_address(target)
        for net in ALLOWED_SCAN_NETWORKS:
            if ip in net:
                return target
        raise ValidationError(
            f"{target} is not an authorised scan target. This build only scans "
            "loopback, your own VirtualBox lab networks, and scanme.nmap.org."
        )
    except ValueError as exc:
        if isinstance(exc, ValidationError):
            raise
        # Not an IP — treat as hostname.
        pass

    if not _HOSTNAME.match(target):
        raise ValidationError("That is not a valid hostname.")
    if target not in ALLOWED_SCAN_HOSTS:
        raise ValidationError(
            f"{target} is not an authorised scan target. This build only scans "
            "loopback, your own VirtualBox lab networks, and scanme.nmap.org."
        )
    return target


def validate_port_spec(ports: str) -> str:
    """Accept '80', '1-1024', '22,80,443'. Nothing else."""
    ports = (ports or "1-1024").strip()
    if not re.fullmatch(r"\d{1,5}(-\d{1,5})?(,\d{1,5}(-\d{1,5})?)*", ports):
        raise ValidationError("Ports must look like 80, 1-1024, or 22,80,443.")
    for part in ports.split(","):
        for n in part.split("-"):
            if not 1 <= int(n) <= 65535:
                raise ValidationError("Port numbers must be between 1 and 65535.")
    return ports


# ── OSINT target rules ───────────────────────────────────────────────────────
#
# sherlock and theHarvester build a profile of a real person or organisation.
# Running them on a third party without a lawful basis is not something a
# student project should do, so the gate here is: the case file must carry a
# signed authorisation record before these endpoints will run at all (enforced
# in app.py), and the input itself must be a plain username or a domain.

_USERNAME = re.compile(r"^[A-Za-z0-9._-]{2,39}$")
_DOMAIN = re.compile(
    r"^(?=.{4,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)"
    r"(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))+$"
)


def validate_username(username: str) -> str:
    username = (username or "").strip()
    if not _USERNAME.match(username):
        raise ValidationError(
            "Username must be 2-39 characters: letters, digits, dot, dash, underscore."
        )
    return username


def validate_domain(domain: str) -> str:
    domain = (domain or "").strip().lower().rstrip(".")
    if not _DOMAIN.match(domain):
        raise ValidationError("Enter a valid domain, e.g. example.com")
    return domain
