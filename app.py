"""
app.py — Deepfake Defence Flask backend
Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import os
import uuid
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from detector import predict_image, predict_video

# ── Config ────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER  = "uploads"
ALLOWED_IMAGE  = {"png", "jpg", "jpeg", "webp", "bmp"}
ALLOWED_VIDEO  = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_CONTENT_MB = 100

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]    = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


def save_file(file):
    ext      = file.filename.rsplit(".", 1)[1].lower()
    name     = f"{uuid.uuid4().hex}.{ext}"
    path     = os.path.join(app.config["UPLOAD_FOLDER"], name)
    file.save(path)
    return path


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyse/image", methods=["POST"])
def analyse_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400
    if not allowed(file.filename, ALLOWED_IMAGE):
        return jsonify({"error": f"Unsupported format. Allowed: {ALLOWED_IMAGE}"}), 400

    path = save_file(file)
    try:
        pil    = Image.open(path)
        result = predict_image(pil)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(path)

    return jsonify(result)


@app.route("/analyse/video", methods=["POST"])
def analyse_video():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400
    if not allowed(file.filename, ALLOWED_VIDEO):
        return jsonify({"error": f"Unsupported format. Allowed: {ALLOWED_VIDEO}"}), 400

    path = save_file(file)
    try:
        result = predict_video(path, sample_every=15)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)

    return jsonify(result)


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🛡️  Deepfake Defence — starting on http://127.0.0.1:5000\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
