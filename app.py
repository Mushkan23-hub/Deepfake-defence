"""
app.py — Deepfake Defence / Digital Forensic Investigation Platform

Run with:  python app.py
Then open: http://127.0.0.1:5000

What changed from the original app.py:
  - It actually imports the modules that were sitting unused: database,
    forensics, audio_detector, pdf_report.
  - Login / signup / sessions exist, so the dashboard and scan history work.
  - Uploads are kept (not deleted immediately) under a random name so a scan
    can be re-run through Kali tools without re-uploading.
  - Case management, evidence, and an audit timeline.
  - Kali tool routes, all of which go through security.py first.
  - debug=True is gone. It gives any visitor a Python shell on your machine.
"""

import io
import os
import json
import uuid
import zipfile
from datetime import datetime, timezone
from functools import wraps

import bcrypt
from dotenv import load_dotenv
from flask import (Flask, request, jsonify, render_template, redirect,
                   session, url_for, flash, send_file, abort)
from PIL import Image

import database as db
import kali_client
from security import (ValidationError, ALLOWED_IMAGE, ALLOWED_VIDEO,
                      ALLOWED_AUDIO, ALLOWED_FORENSIC, MAX_UPLOAD_MB,
                      check_extension, safe_stored_path)
from detector import predict_image, predict_video
from audio_detector import predict_audio
from forensics import run_exiftool, run_ela, run_ffprobe
from pdf_report import generate_report

load_dotenv()

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.secret_key = os.getenv("SECRET_KEY")

if not app.secret_key:
    raise SystemExit(
        "SECRET_KEY is not set. Create a .env file with:\n"
        "  SECRET_KEY=<run: python -c \"import secrets;print(secrets.token_hex(32))\">"
    )

# Session cookie hardening. SECURE stays off for local http:// development;
# set COOKIE_SECURE=1 in .env once you deploy behind HTTPS.
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "0") == "1",
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _asset_version(filename):
    """
    Mtime of a static file, used as a cache-busting query param. Without this,
    browsers (and sometimes Flask's own conditional-request handling) can keep
    serving an old cached copy of style.css/app.js after you've replaced the
    file on disk, even after a normal refresh — the URL never changes, so the
    browser has no reason to re-fetch it.
    """
    path = os.path.join(app.static_folder, filename)
    try:
        return int(os.path.getmtime(path))
    except OSError:
        return 0


app.jinja_env.globals["asset_version"] = _asset_version


# ── Helpers ──────────────────────────────────────────────────────────────────

def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Not signed in."}), 401
            flash("Please sign in to access that page.", "error")
            return redirect(url_for("login_page", next=request.path))
        return view(*args, **kwargs)
    return wrapper


def _safe_next(raw):
    """Only allow same-site redirect targets, never an absolute or protocol-relative URL."""
    if raw and raw.startswith("/") and not raw.startswith("//"):
        return raw
    return url_for("index")


def save_upload(file_storage, allowed_set):
    """Store under a random name. The user's filename is never used as a path."""
    ext = check_extension(file_storage.filename or "", allowed_set)
    stored = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_FOLDER, stored)
    file_storage.save(path)
    return stored, path


def current_case_id():
    """Optional case to attach results to. Absent = ad-hoc scan."""
    case_id = request.form.get("case_id") or (request.get_json(silent=True) or {}).get("case_id")
    if not case_id:
        return None
    if not db.get_case(case_id, session["user_id"]):
        raise ValidationError("Case not found.")
    return case_id


def require_authorised_case(case_id):
    """
    Gate for anything that reaches out to a third party. A case must exist and
    carry an authorisation record before nmap / sherlock / theHarvester runs.
    """
    if not case_id:
        raise ValidationError(
            "Select a case first. Tools that contact external systems must be "
            "attached to a case with a recorded authorisation."
        )
    case = db.get_case(case_id, session["user_id"])
    if not case:
        raise ValidationError("Case not found.")
    if not case.get("authorisation"):
        raise ValidationError(
            "This case has no authorisation record. Add one (authority, "
            "reference, scope) before running external tools."
        )
    return case


@app.errorhandler(ValidationError)
def handle_validation(exc):
    return jsonify({"error": str(exc)}), 400


@app.errorhandler(413)
def handle_too_large(_):
    return jsonify({"error": f"File is larger than {MAX_UPLOAD_MB} MB."}), 413


# ── Auth ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    user_id = session.get("user_id")
    stats = db.get_scan_stats(user_id) if user_id else None
    recent_scans = db.get_user_scans(user_id, limit=5) if user_id else []
    return render_template("home.html", active="home", stats=stats, recent_scans=recent_scans)


@app.route("/login", methods=["GET"])
def login_page():
    if "user_id" in session:
        return redirect(url_for("index"))
    return render_template("login.html", next=request.args.get("next", ""))


@app.route("/signup", methods=["GET"])
def signup_page():
    if "user_id" in session:
        return redirect(url_for("index"))
    return render_template("signup.html")


@app.route("/signup", methods=["POST"])
def signup():
    username = (request.form.get("username") or "").strip()
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    if not (3 <= len(username) <= 32):
        flash("Username must be 3-32 characters.", "error")
        return redirect(url_for("signup_page"))
    if "@" not in email or len(email) < 5:
        flash("Enter a valid email address.", "error")
        return redirect(url_for("signup_page"))
    if len(password) < 8:
        flash("Password must be at least 8 characters.", "error")
        return redirect(url_for("signup_page"))

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user_id, err = db.create_user(username, email, pw_hash)
    if err:
        flash(err, "error")
        return redirect(url_for("signup_page"))

    session.clear()
    session["user_id"] = user_id
    session["username"] = username
    return redirect(url_for("index"))


@app.route("/login", methods=["POST"])
def login():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    next_url = _safe_next(request.form.get("next"))
    user = db.get_user_by_email(email)

    # Same message either way, so the form can't be used to enumerate accounts.
    if not user or not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        flash("Incorrect email or password.", "error")
        return redirect(url_for("login_page", next=next_url))

    session.clear()
    session["user_id"] = str(user["_id"])
    session["username"] = user["username"]
    return redirect(next_url)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ── Pages ────────────────────────────────────────────────────────────────────

@app.route("/activity")
@login_required
def activity():
    user_id = session["user_id"]
    return render_template(
        "activity.html",
        active="activity",
        stats=db.get_scan_stats(user_id),
        timeline=db.get_scan_timeline(user_id),
        scans=db.get_user_scans(user_id),
    )


@app.route("/dashboard")
@login_required
def dashboard():
    """Old URL, kept working — redirects to the renamed page."""
    return redirect(url_for("activity"))


@app.route("/cases")
@login_required
def cases_page():
    return render_template("cases.html", active="cases")


@app.route("/profile")
@login_required
def profile_page():
    user = db.get_user_by_id(session["user_id"])
    if not user:
        abort(404)
    return render_template(
        "profile.html", active="profile", user=user,
        stats=db.get_scan_stats(session["user_id"]),
    )


@app.route("/profile", methods=["POST"])
@login_required
def update_profile():
    new_username = (request.form.get("username") or "").strip()
    full_name = (request.form.get("full_name") or "").strip()
    bio = (request.form.get("bio") or "").strip()

    if not (3 <= len(new_username) <= 32):
        flash("Username must be 3-32 characters.", "error")
        return redirect(url_for("profile_page"))

    if new_username != session.get("username"):
        ok, err = db.update_username(session["user_id"], new_username)
        if not ok:
            flash(err, "error")
            return redirect(url_for("profile_page"))
        session["username"] = new_username

    db.update_user_profile(session["user_id"], full_name, bio)
    flash("Profile updated.", "info")
    return redirect(url_for("profile_page"))


@app.route("/settings")
@login_required
def settings_page():
    return render_template(
        "settings.html", active="settings",
        kali_online=kali_client.is_online(), kali_url=kali_client.AGENT_URL,
    )


@app.route("/settings/password", methods=["POST"])
@login_required
def change_password():
    current = request.form.get("current_password") or ""
    new = request.form.get("new_password") or ""
    confirm = request.form.get("confirm_password") or ""

    user = db.get_user_by_id(session["user_id"])
    if not user or not bcrypt.checkpw(current.encode(), user["password_hash"].encode()):
        flash("Current password is incorrect.", "error")
        return redirect(url_for("settings_page"))
    if len(new) < 8:
        flash("New password must be at least 8 characters.", "error")
        return redirect(url_for("settings_page"))
    if new != confirm:
        flash("New password and confirmation don't match.", "error")
        return redirect(url_for("settings_page"))

    new_hash = bcrypt.hashpw(new.encode(), bcrypt.gensalt()).decode()
    db.update_user_password(session["user_id"], new_hash)
    flash("Password changed.", "info")
    return redirect(url_for("settings_page"))


# ── AI detection tool pages ──────────────────────────────────────────────────

_DETECT_META = {
    "image": {"title": "Image Scan", "accept": "image/*",
              "accept_hint": "PNG, JPG, WEBP, BMP",
              "description": "Upload a photo to check for signs of AI generation or face-swap manipulation."},
    "video": {"title": "Video Scan", "accept": "video/*",
              "accept_hint": "MP4, AVI, MOV, MKV, WEBM",
              "description": "Samples frames across the video and aggregates a deepfake verdict."},
    "audio": {"title": "Audio Scan", "accept": "audio/*",
              "accept_hint": "WAV, MP3, FLAC, OGG, M4A",
              "description": "Analyses voice recordings for signs of synthetic cloning or splicing."},
}


@app.route("/tools/image")
@login_required
def tool_image():
    meta = _DETECT_META["image"]
    return render_template("tool_detect.html", active="tool_image", kind="image",
                           cases=db.get_cases(session["user_id"]), **meta)


@app.route("/tools/video")
@login_required
def tool_video():
    meta = _DETECT_META["video"]
    return render_template("tool_detect.html", active="tool_video", kind="video",
                           cases=db.get_cases(session["user_id"]), **meta)


@app.route("/tools/audio")
@login_required
def tool_audio():
    meta = _DETECT_META["audio"]
    return render_template("tool_detect.html", active="tool_audio", kind="audio",
                           cases=db.get_cases(session["user_id"]), **meta)


# ── Kali file-tool pages ─────────────────────────────────────────────────────

_FILE_TOOL_META = {
    "steghide":        {"title": "Steganography Detection", "binary": "steghide",
                         "description": "Checks whether an image or audio file has hidden data embedded with steghide."},
    "binwalk":         {"title": "Binary Analysis", "binary": "binwalk",
                         "description": "Scans a file for embedded file signatures — firmware images, archives, other files hidden inside it."},
    "binwalk_extract": {"title": "Extract Embedded Files", "binary": "binwalk",
                         "description": "Runs binwalk in extraction mode and lists every file it manages to carve out."},
    "foremost":        {"title": "File Carving", "binary": "foremost",
                         "description": "Recovers files from raw data by signature — useful on disk images or files with damaged headers."},
    "exiftool":        {"title": "Metadata Extraction", "binary": "exiftool",
                         "description": "Pulls every EXIF/XMP/IPTC metadata field from an image, video or document."},
    "strings":         {"title": "Printable Strings", "binary": "strings",
                         "description": "Extracts human-readable text embedded in a binary file — useful for spotting URLs, paths or credentials."},
    "tshark_summary":  {"title": "PCAP Protocol Hierarchy", "binary": "tshark",
                         "description": "Summarises the protocol breakdown of a captured network traffic file (.pcap/.pcapng)."},
    "tshark_convs":    {"title": "PCAP IP Conversations", "binary": "tshark",
                         "description": "Lists every IP-to-IP conversation found in a captured network traffic file."},
}


@app.route("/tools/file/<tool>")
@login_required
def tool_file_page(tool):
    meta = _FILE_TOOL_META.get(tool)
    if not meta:
        abort(404)
    tools_status = kali_client.available_tools()
    return render_template(
        "tool_file.html", active=tool, tool=tool,
        title=meta["title"], description=meta["description"], tool_binary=meta["binary"],
        cases=db.get_cases(session["user_id"]),
        kali_online=kali_client.is_online(),
        tool_installed=tools_status.get(tool, False),
    )


@app.route("/tools/nmap")
@login_required
def tool_nmap():
    return render_template("tool_network.html", active="nmap",
                           cases=db.get_cases(session["user_id"]),
                           kali_online=kali_client.is_online())


_OSINT_META = {
    "sherlock": {"title": "Username Search (Sherlock)",
                 "description": "Searches 300+ sites for a given username. Requires a case with a recorded authorisation.",
                 "input_label": "Username", "input_placeholder": "e.g. johndoe123"},
    "theharvester": {"title": "Domain Harvesting (theHarvester)",
                      "description": "Gathers emails, subdomains and hosts for a domain from public sources. Requires a case with a recorded authorisation.",
                      "input_label": "Domain", "input_placeholder": "e.g. example.com"},
}


@app.route("/tools/osint/<kind>")
@login_required
def tool_osint_page(kind):
    meta = _OSINT_META.get(kind)
    if not meta:
        abort(404)
    return render_template("tool_osint.html", active=kind, kind=kind,
                           cases=db.get_cases(session["user_id"]),
                           kali_online=kali_client.is_online(), **meta)


# ── Detection API ────────────────────────────────────────────────────────────

@app.route("/api/analyse/image", methods=["POST"])
@login_required
def analyse_image():
    if "file" not in request.files or not request.files["file"].filename:
        raise ValidationError("No file provided.")
    case_id = current_case_id()
    stored, path = save_upload(request.files["file"], ALLOWED_IMAGE)

    try:
        result = predict_image(Image.open(path))
        result["forensics"] = {
            "exiftool": run_exiftool(path),
            "ela": run_ela(path),
        }
        result["stored_name"] = stored
        result["hashes"] = kali_client.compute_hashes(path, stored).get("hashes", {})
        scan_id = db.save_scan(session["user_id"], request.files["file"].filename,
                               "image", result, case_id)
        result["scan_id"] = scan_id
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 500


@app.route("/api/analyse/video", methods=["POST"])
@login_required
def analyse_video():
    if "file" not in request.files or not request.files["file"].filename:
        raise ValidationError("No file provided.")
    case_id = current_case_id()
    stored, path = save_upload(request.files["file"], ALLOWED_VIDEO)

    try:
        result = predict_video(path, sample_every=15)
        if "error" in result:
            return jsonify(result), 400
        result["forensics"] = {
            "exiftool": run_exiftool(path),
            "ffprobe": run_ffprobe(path),
        }
        result["stored_name"] = stored
        result["hashes"] = kali_client.compute_hashes(path, stored).get("hashes", {})
        result["scan_id"] = db.save_scan(session["user_id"],
                                         request.files["file"].filename,
                                         "video", result, case_id)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 500


@app.route("/api/analyse/audio", methods=["POST"])
@login_required
def analyse_audio():
    if "file" not in request.files or not request.files["file"].filename:
        raise ValidationError("No file provided.")
    case_id = current_case_id()
    stored, path = save_upload(request.files["file"], ALLOWED_AUDIO)

    try:
        result = predict_audio(path)
        if "error" in result:
            return jsonify(result), 400
        result["forensics"] = {**result.get("forensics", {}), "exiftool": run_exiftool(path)}
        result["stored_name"] = stored
        result["hashes"] = kali_client.compute_hashes(path, stored).get("hashes", {})
        result["scan_id"] = db.save_scan(session["user_id"],
                                         request.files["file"].filename,
                                         "audio", result, case_id)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 500


@app.route("/api/scan/<scan_id>")
@login_required
def get_scan(scan_id):
    scan = db.get_scan_by_id(scan_id, session["user_id"])
    if not scan:
        return jsonify({"error": "Scan not found."}), 404
    scan["timestamp"] = scan["timestamp"].isoformat()
    return jsonify(scan)


@app.route("/report/<scan_id>")
@login_required
def report(scan_id):
    scan = db.get_scan_by_id(scan_id, session["user_id"])
    if not scan:
        abort(404)
    pdf = generate_report(scan, session.get("username", "investigator"))
    return send_file(io.BytesIO(pdf), mimetype="application/pdf",
                     as_attachment=True,
                     download_name=f"forensic-report-{scan_id}.pdf")


# ── Case management ──────────────────────────────────────────────────────────

@app.route("/api/cases", methods=["GET"])
@login_required
def list_cases():
    cases = db.get_cases(session["user_id"])
    for c in cases:
        c["created_at"] = c["created_at"].isoformat()
        c["updated_at"] = c["updated_at"].isoformat()
        if c.get("authorisation"):
            c["authorisation"]["recorded_at"] = c["authorisation"]["recorded_at"].isoformat()
    return jsonify(cases)


@app.route("/api/cases", methods=["POST"])
@login_required
def new_case():
    body = request.get_json(silent=True) or request.form
    title = (body.get("title") or "").strip()
    if not (3 <= len(title) <= 120):
        raise ValidationError("Case title must be 3-120 characters.")
    case_id = db.create_case(session["user_id"], title,
                             body.get("description") or "")
    return jsonify({"case_id": case_id}), 201


@app.route("/api/cases/<case_id>")
@login_required
def case_detail(case_id):
    case = db.get_case(case_id, session["user_id"])
    if not case:
        return jsonify({"error": "Case not found."}), 404
    case["created_at"] = case["created_at"].isoformat()
    case["updated_at"] = case["updated_at"].isoformat()
    if case.get("authorisation"):
        case["authorisation"]["recorded_at"] = case["authorisation"]["recorded_at"].isoformat()

    scans = db.get_case_scans(case_id, session["user_id"])
    for s in scans:
        s["timestamp"] = s["timestamp"].isoformat()
    evidence = db.get_case_evidence(case_id, session["user_id"])
    for e in evidence:
        e["created_at"] = e["created_at"].isoformat()

    return jsonify({
        "case": case,
        "scans": scans,
        "evidence": evidence,
        "timeline": db.get_case_timeline(case_id, session["user_id"]),
    })


@app.route("/api/cases/<case_id>/authorisation", methods=["POST"])
@login_required
def set_authorisation(case_id):
    if not db.get_case(case_id, session["user_id"]):
        return jsonify({"error": "Case not found."}), 404
    body = request.get_json(silent=True) or request.form
    authority = (body.get("authority") or "").strip()
    reference = (body.get("reference") or "").strip()
    scope = (body.get("scope") or "").strip()
    if not authority or not reference or not scope:
        raise ValidationError(
            "All three fields are required: who authorised this, the written "
            "reference (letter/ticket/consent form), and what it covers."
        )
    record = db.set_case_authorisation(case_id, session["user_id"],
                                       authority, reference, scope)
    record["recorded_at"] = record["recorded_at"].isoformat()
    return jsonify(record), 201


@app.route("/api/cases/<case_id>/close", methods=["POST"])
@login_required
def close_case(case_id):
    if not db.get_case(case_id, session["user_id"]):
        return jsonify({"error": "Case not found."}), 404
    db.close_case(case_id, session["user_id"])
    return jsonify({"status": "closed"})


# ── Kali integration ─────────────────────────────────────────────────────────

@app.route("/api/kali/status")
@login_required
def kali_status():
    online = kali_client.is_online()
    return jsonify({
        "online": online,
        "agent_url": kali_client.AGENT_URL,
        "tools": kali_client.available_tools() if online else {},
        "file_tools": kali_client.FILE_TOOLS,
    })


@app.route("/api/kali/file/<tool>", methods=["POST"])
@login_required
def kali_file_tool(tool):
    """
    Run a file-based Kali tool. Two ways to supply the file:
      - upload a new one as 'file'
      - reference an already-stored upload by 'stored_name'
    """
    case_id = current_case_id()
    cleanup = False

    if "file" in request.files and request.files["file"].filename:
        original = request.files["file"].filename
        stored, path = save_upload(request.files["file"], ALLOWED_FORENSIC)
        cleanup = False   # keep it; the case may need it again
    else:
        stored = (request.form.get("stored_name") or "").strip()
        path = safe_stored_path(UPLOAD_FOLDER, stored)
        original = stored

    result = kali_client.run_file_tool(tool, path, original)
    result["tool"] = tool

    if case_id:
        db.save_evidence(case_id, session["user_id"], tool, original, result)
    return jsonify(result)


@app.route("/api/kali/network/nmap", methods=["POST"])
@login_required
def kali_nmap():
    body = request.get_json(silent=True) or request.form
    case_id = current_case_id()
    require_authorised_case(case_id)

    target = body.get("target") or ""
    ports = body.get("ports") or "1-1024"
    result = kali_client.scan_network(target, ports)
    db.save_evidence(case_id, session["user_id"], "nmap", target, result)
    return jsonify(result)


@app.route("/api/kali/osint/sherlock", methods=["POST"])
@login_required
def kali_sherlock():
    body = request.get_json(silent=True) or request.form
    case_id = current_case_id()
    require_authorised_case(case_id)

    username = body.get("username") or ""
    result = kali_client.osint_username(username)
    db.save_evidence(case_id, session["user_id"], "sherlock", username, result)
    return jsonify(result)


@app.route("/api/kali/osint/theharvester", methods=["POST"])
@login_required
def kali_theharvester():
    body = request.get_json(silent=True) or request.form
    case_id = current_case_id()
    require_authorised_case(case_id)

    domain = body.get("domain") or ""
    result = kali_client.osint_domain(domain)
    db.save_evidence(case_id, session["user_id"], "theHarvester", domain, result)
    return jsonify(result)


# ── Evidence export ──────────────────────────────────────────────────────────

@app.route("/api/cases/<case_id>/export")
@login_required
def export_case(case_id):
    """Package the whole case as a ZIP: JSON bundle, tool output, PDF reports."""
    user_id = session["user_id"]
    case = db.get_case(case_id, user_id)
    if not case:
        abort(404)

    scans = db.get_case_scans(case_id, user_id)
    evidence = db.get_case_evidence(case_id, user_id)
    timeline = db.get_case_timeline(case_id, user_id)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        bundle = {
            "case": case,
            "scans": scans,
            "evidence": evidence,
            "timeline": timeline,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "exported_by": session.get("username"),
        }
        zf.writestr("case.json", json.dumps(bundle, default=str, indent=2))

        lines = [f"CASE: {case['title']}", f"Status: {case['status']}", ""]
        auth = case.get("authorisation")
        lines.append("AUTHORISATION")
        if auth:
            lines += [f"  Authority: {auth['authority']}",
                      f"  Reference: {auth['reference']}",
                      f"  Scope:     {auth['scope']}", ""]
        else:
            lines += ["  None recorded. No external-facing tools were run.", ""]
        lines.append("TIMELINE")
        lines += [f"  {e['at']}  {e['action']:<22} {e['detail']}" for e in timeline]
        zf.writestr("chain-of-custody.txt", "\n".join(lines))

        for ev in evidence:
            name = f"evidence/{ev['created_at']:%Y%m%d-%H%M%S}-{ev['tool']}.txt"
            body = (f"Tool:    {ev['tool']}\nSubject: {ev['subject']}\n"
                    f"Command: {ev['command']}\nOK:      {ev['ok']}\n"
                    f"Hashes:  {json.dumps(ev.get('hashes', {}))}\n\n"
                    f"--- stdout ---\n{ev['stdout']}\n\n--- stderr ---\n{ev['stderr']}\n")
            zf.writestr(name, body)

        for s in scans:
            try:
                pdf = generate_report(s, session.get("username", "investigator"))
                zf.writestr(f"reports/scan-{s['_id']}.pdf", pdf)
            except Exception as exc:
                zf.writestr(f"reports/scan-{s['_id']}.ERROR.txt", str(exc))

    buf.seek(0)
    db.log_action(case_id, user_id, "case.exported", "Case exported as ZIP.")
    safe_title = "".join(c if c.isalnum() else "-" for c in case["title"])[:40]
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"case-{safe_title}-{case_id}.zip")


# ── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Deepfake Defence — http://127.0.0.1:5000")
    print(f"  Kali agent: {kali_client.AGENT_URL} "
          f"({'online' if kali_client.is_online() else 'offline'})\n")
    # debug=False deliberately: Werkzeug's debugger is a remote shell.
    app.run(debug=False, host="127.0.0.1", port=5000)
