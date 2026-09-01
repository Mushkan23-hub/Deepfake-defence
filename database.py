"""
database.py — MongoDB interface
"""
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

client    = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
db        = client[os.getenv("MONGO_DB", "deepfake_defence")]
users_col = db["users"]
scans_col = db["scans"]


def create_user(username, email, password_hash):
    if users_col.find_one({"email": email}):
        return None, "Email already registered."
    if users_col.find_one({"username": username}):
        return None, "Username already taken."
    doc = {"username": username, "email": email,
           "password_hash": password_hash, "created_at": datetime.utcnow()}
    result = users_col.insert_one(doc)
    return str(result.inserted_id), None


def get_user_by_email(email):
    return users_col.find_one({"email": email})


def get_user_by_id(user_id):
    try:
        return users_col.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None


def save_scan(user_id, filename, file_type, result):
    doc = {
        "user_id":     user_id,
        "filename":    filename,
        "file_type":   file_type,
        "verdict":     result["verdict"],
        "confidence":  result["confidence"],
        "is_fake":     result["is_fake"],
        "ai_scores":   result.get("scores", {}),
        "forensics":   result.get("forensics", {}),
        "explanation": result.get("explanation", []),
        "timestamp":   datetime.utcnow(),
    }
    return str(scans_col.insert_one(doc).inserted_id)


def get_user_scans(user_id, limit=50):
    cursor = scans_col.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
    scans  = []
    for doc in cursor:
        doc["_id"]       = str(doc["_id"])
        doc["timestamp"] = doc["timestamp"].strftime("%d %b %Y, %H:%M")
        scans.append(doc)
    return scans


def get_scan_stats(user_id):
    total = scans_col.count_documents({"user_id": user_id})
    fake  = scans_col.count_documents({"user_id": user_id, "is_fake": True})
    by_type = {t: scans_col.count_documents({"user_id": user_id, "file_type": t})
               for t in ["image", "video", "audio"]}
    return {"total": total, "fake": fake, "real": total - fake, "by_type": by_type}


def get_scan_timeline(user_id, days=30):
    cutoff = datetime.utcnow() - timedelta(days=days)
    cursor = scans_col.find(
        {"user_id": user_id, "timestamp": {"$gte": cutoff}},
        {"timestamp": 1, "is_fake": 1, "_id": 0}
    ).sort("timestamp", 1)
    timeline = {}
    for doc in cursor:
        day = doc["timestamp"].strftime("%d %b")
        if day not in timeline:
            timeline[day] = {"total": 0, "fake": 0}
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
