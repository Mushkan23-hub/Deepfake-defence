"""
database.py — MongoDB interface.

Changes from the original:
  - datetime.utcnow() is deprecated in Python 3.12; uses timezone-aware now().
  - Indexes are created at import, so dashboard queries stay fast as scans grow.
  - New collections: cases, evidence, audit.
  - Every case action is written to an append-only audit trail. That trail is
    what turns this from "a tool that ran" into something defensible — a
    forensic report is only worth anything if you can show who did what, when,
    and under what authority.
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from bson.objectid import ObjectId
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
db = client[os.getenv("MONGO_DB", "deepfake_defence")]

users_col    = db["users"]
scans_col    = db["scans"]
cases_col    = db["cases"]
evidence_col = db["evidence"]
audit_col    = db["audit"]


def _now():
    return datetime.now(timezone.utc)


def ensure_indexes():
    users_col.create_index([("email", ASCENDING)], unique=True)
    users_col.create_index([("username", ASCENDING)], unique=True)
    scans_col.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
    scans_col.create_index([("case_id", ASCENDING)])
    cases_col.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    evidence_col.create_index([("case_id", ASCENDING), ("created_at", ASCENDING)])
    audit_col.create_index([("case_id", ASCENDING), ("at", ASCENDING)])


try:
    ensure_indexes()
except Exception as exc:          # Mongo not running yet — app.py reports it
    print(f"[database] index setup skipped: {exc}")


# ── Users ────────────────────────────────────────────────────────────────────

def create_user(username, email, password_hash):
    if users_col.find_one({"email": email}):
        return None, "Email already registered."
    if users_col.find_one({"username": username}):
        return None, "Username already taken."
    doc = {"username": username, "email": email,
           "password_hash": password_hash, "created_at": _now()}
    return str(users_col.insert_one(doc).inserted_id), None


def get_user_by_email(email):
    return users_col.find_one({"email": email})


def get_user_by_id(user_id):
    try:
        return users_col.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None


# ── Scans ────────────────────────────────────────────────────────────────────

def save_scan(user_id, filename, file_type, result, case_id=None):
    doc = {
        "user_id":     user_id,
        "case_id":     case_id,
        "filename":    filename,
        "file_type":   file_type,
        "verdict":     result.get("verdict", "UNKNOWN"),
        "confidence":  result.get("confidence", 0),
        "is_fake":     result.get("is_fake", False),
        "ai_scores":   result.get("scores", {}),
        "forensics":   result.get("forensics", {}),
        "explanation": result.get("explanation", []),
        "hashes":      result.get("hashes", {}),
        "timestamp":   _now(),
    }
    scan_id = str(scans_col.insert_one(doc).inserted_id)
    if case_id:
        log_action(case_id, user_id, "scan.created",
                   f"{file_type} scan of {filename} -> {doc['verdict']}",
                   {"scan_id": scan_id})
    return scan_id


def get_user_scans(user_id, limit=50):
    cursor = scans_col.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
    scans = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["timestamp"] = doc["timestamp"].strftime("%d %b %Y, %H:%M")
        scans.append(doc)
    return scans


def get_scan_stats(user_id):
    total = scans_col.count_documents({"user_id": user_id})
    fake = scans_col.count_documents({"user_id": user_id, "is_fake": True})
    by_type = {t: scans_col.count_documents({"user_id": user_id, "file_type": t})
               for t in ["image", "video", "audio"]}
    return {"total": total, "fake": fake, "real": total - fake, "by_type": by_type}


def get_scan_timeline(user_id, days=30):
    cutoff = _now() - timedelta(days=days)
    cursor = scans_col.find(
        {"user_id": user_id, "timestamp": {"$gte": cutoff}},
        {"timestamp": 1, "is_fake": 1, "_id": 0}
    ).sort("timestamp", 1)
    timeline = {}
    for doc in cursor:
        day = doc["timestamp"].strftime("%d %b")
        timeline.setdefault(day, {"total": 0, "fake": 0})
        timeline[day]["total"] += 1
        if doc["is_fake"]:
            timeline[day]["fake"] += 1
    return timeline


def get_scan_by_id(scan_id, user_id):
    try:
        doc = scans_col.find_one({"_id": ObjectId(scan_id), "user_id": user_id})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception:
        return None


# ── Cases ────────────────────────────────────────────────────────────────────

def create_case(user_id, title, description=""):
    doc = {
        "user_id": user_id,
        "title": title.strip()[:120],
        "description": description.strip()[:2000],
        "status": "open",
        "authorisation": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    case_id = str(cases_col.insert_one(doc).inserted_id)
    log_action(case_id, user_id, "case.opened", f"Case opened: {doc['title']}")
    return case_id


def get_cases(user_id):
    out = []
    for doc in cases_col.find({"user_id": user_id}).sort("created_at", -1):
        doc["_id"] = str(doc["_id"])
        doc["evidence_count"] = evidence_col.count_documents({"case_id": doc["_id"]})
        doc["scan_count"] = scans_col.count_documents({"case_id": doc["_id"]})
        out.append(doc)
    return out


def get_case(case_id, user_id):
    try:
        doc = cases_col.find_one({"_id": ObjectId(case_id), "user_id": user_id})
    except Exception:
        return None
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


def set_case_authorisation(case_id, user_id, authority, reference, scope):
    """
    Record the lawful basis for running tools that touch third parties.
    Nothing in the OSINT/network section runs until this exists.
    """
    record = {
        "authority": authority.strip()[:200],
        "reference": reference.strip()[:200],
        "scope": scope.strip()[:1000],
        "recorded_by": user_id,
        "recorded_at": _now(),
    }
    cases_col.update_one(
        {"_id": ObjectId(case_id), "user_id": user_id},
        {"$set": {"authorisation": record, "updated_at": _now()}},
    )
    log_action(case_id, user_id, "case.authorised",
               f"Authorisation recorded: {record['authority']} ({record['reference']})")
    return record


def close_case(case_id, user_id):
    cases_col.update_one({"_id": ObjectId(case_id), "user_id": user_id},
                         {"$set": {"status": "closed", "updated_at": _now()}})
    log_action(case_id, user_id, "case.closed", "Case closed.")


# ── Evidence (results of Kali tool runs) ─────────────────────────────────────

def save_evidence(case_id, user_id, tool, subject, result):
    doc = {
        "case_id": case_id,
        "user_id": user_id,
        "tool": tool,
        "subject": subject,
        "ok": bool(result.get("ok")),
        "command": result.get("command", ""),
        "stdout": (result.get("stdout") or "")[:200_000],
        "stderr": (result.get("stderr") or "")[:8_000],
        "hashes": result.get("hashes", {}),
        "artifacts": result.get("artifacts", []),
        "error": result.get("error"),
        "created_at": _now(),
    }
    ev_id = str(evidence_col.insert_one(doc).inserted_id)
    log_action(case_id, user_id, "evidence.collected",
               f"{tool} run against {subject}", {"evidence_id": ev_id})
    return ev_id


def get_case_evidence(case_id, user_id):
    out = []
    for doc in evidence_col.find({"case_id": case_id, "user_id": user_id}).sort("created_at", 1):
        doc["_id"] = str(doc["_id"])
        out.append(doc)
    return out


def get_case_scans(case_id, user_id):
    out = []
    for doc in scans_col.find({"case_id": case_id, "user_id": user_id}).sort("timestamp", 1):
        doc["_id"] = str(doc["_id"])
        out.append(doc)
    return out


# ── Audit trail ──────────────────────────────────────────────────────────────

def log_action(case_id, user_id, action, detail, meta=None):
    audit_col.insert_one({
        "case_id": case_id,
        "user_id": user_id,
        "action": action,
        "detail": detail,
        "meta": meta or {},
        "at": _now(),
    })


def get_case_timeline(case_id, user_id):
    out = []
    for doc in audit_col.find({"case_id": case_id, "user_id": user_id}).sort("at", 1):
        out.append({
            "at": doc["at"].strftime("%d %b %Y, %H:%M:%S UTC"),
            "action": doc["action"],
            "detail": doc["detail"],
            "meta": doc.get("meta", {}),
        })
    return out
