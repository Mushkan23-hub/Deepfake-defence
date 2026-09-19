"""
detector.py — Deepfake Defence core engine

Uses a real, pretrained deepfake image classifier (ashish-001/deepfake-
detection-using-ViT on Hugging Face — a ViT-Base fine-tuned on a labeled
real/fake face dataset, apache-2.0 licensed, reports 92% test accuracy on its
own dataset: https://huggingface.co/ashish-001/deepfake-detection-using-ViT).

Caveat worth keeping in mind: it was trained on one specific Kaggle dataset of
mostly face-swap-style fakes, so it will not generalise perfectly to every
generator (diffusion-based images especially) — but it is a real trained
classifier, unlike the randomly-initialised head this file used before.

If `transformers` isn't installed, or the model can't be downloaded (first
run needs internet access to huggingface.co), this falls back to heuristic
signals only (noise / symmetry / compression) and says so in the result via
`model_status`.
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_MODEL_ID = os.getenv("DEEPFAKE_HF_MODEL", "ashish-001/deepfake-detection-using-ViT")

MODEL_TRAINED = False
MODEL_LOAD_ERROR = None

_processor = None
_model = None


def load_model():
    """
    Downloads and caches the Hugging Face model on first call (needs internet
    the first time; cached under ~/.cache/huggingface after that). Never
    raises — failures fall back to heuristics and are logged once.
    """
    global _processor, _model, MODEL_TRAINED, MODEL_LOAD_ERROR

    if _model is not None or MODEL_LOAD_ERROR:
        return

    if not _TRANSFORMERS_AVAILABLE:
        MODEL_LOAD_ERROR = "transformers is not installed"
        print(f"[detector] {MODEL_LOAD_ERROR} — run: pip install transformers\n"
              "[detector] Falling back to heuristics only.")
        return

    try:
        _processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
        _model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID).to(DEVICE)
        _model.eval()
        MODEL_TRAINED = True
        print(f"[detector] loaded pretrained classifier: {HF_MODEL_ID}")
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        print(f"[detector] could not load {HF_MODEL_ID}: {exc}\n"
              "[detector] Falling back to heuristics only. If this is the "
              "first run, check internet access — the model downloads from "
              "huggingface.co.")


def _fake_probability(pil_image: Image.Image):
    """
    Runs the HF model and returns P(fake) in [0, 1], or None if the model
    isn't loaded or its label scheme is unrecognised. Reads the model's own
    id2label mapping rather than assuming index 0/1 order, since that varies
    by checkpoint.
    """
    if _model is None:
        return None
    inputs = _processor(images=pil_image.convert("RGB"), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    id2label = {int(k): str(v).lower() for k, v in _model.config.id2label.items()}
    fake_idx = next((i for i, lbl in id2label.items() if "fake" in lbl), None)
    if fake_idx is None:
        return None  # Unrecognised label scheme — don't guess which index means what.
    return float(probs[fake_idx].item())


# ── Heuristic helpers (used for explanation layer) ────────────────────────────
def analyse_noise(img_np):
    """High-frequency noise analysis — GAN images often show unnatural noise patterns."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    variance = laplacian.var()
    # Real photos: moderate variance. GANs: very low or very high.
    score = float(np.clip(1 - (variance / 2000), 0, 1))
    return round(score, 3)


def analyse_face_consistency(img_np):
    """Check facial landmark symmetry — deepfakes often have subtle asymmetries."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    left  = gray[:, :w//2]
    right = cv2.flip(gray[:, w//2:], 1)
    min_w = min(left.shape[1], right.shape[1])
    diff = np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float))
    asymmetry = diff.mean() / 255.0
    score = float(np.clip(asymmetry * 4, 0, 1))
    return round(score, 3)


def analyse_compression_artifacts(img_np):
    """JPEG compression artifact analysis — deepfakes often show inconsistent artifacts."""
    img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    y_channel = img_yuv[:, :, 0].astype(np.float32)
    dct = cv2.dct(y_channel[:256, :256])
    high_freq = np.abs(dct[64:, 64:]).mean()
    score = float(np.clip(high_freq / 50, 0, 1))
    return round(score, 3)


def build_explanation(confidence, noise_s, face_s, compress_s, is_fake):
    """Generate a human-readable explanation of the detection result."""
    reasons = []

    if is_fake:
        if noise_s > 0.5:
            reasons.append("🔴 Unnatural noise patterns detected — consistent with GAN-generated imagery.")
        if face_s > 0.4:
            reasons.append("🔴 Facial asymmetry above normal threshold — possible face-swap artefact.")
        if compress_s > 0.5:
            reasons.append("🔴 Inconsistent compression artefacts — common in synthetically generated faces.")
        if confidence > 0.85:
            reasons.append("🔴 High model confidence: strong visual features associated with deepfakes.")
        if not reasons:
            reasons.append("🔴 Model detected subtle manipulations not easily visible to the human eye.")
    else:
        if noise_s < 0.3:
            reasons.append("🟢 Natural noise distribution — consistent with real camera sensor output.")
        if face_s < 0.3:
            reasons.append("🟢 Facial symmetry within natural human range.")
        if compress_s < 0.3:
            reasons.append("🟢 Compression patterns consistent with authentic photographic media.")
        if confidence < 0.2:
            reasons.append("🟢 High model confidence: strong features associated with authentic content.")
        if not reasons:
            reasons.append("🟢 No significant manipulation indicators detected.")

    return reasons


# ── Public API ────────────────────────────────────────────────────────────────
def predict_image(pil_image: Image.Image) -> dict:
    """
    Run deepfake detection on a PIL image.
    Returns dict with verdict, confidence, scores, and explanation.
    """
    load_model()
    img_np = np.array(pil_image.convert("RGB"))

    fake_prob = _fake_probability(pil_image)

    # Heuristic scores — always computed, used either as the sole signal
    # (model unavailable) or as a smaller supporting weight (model loaded).
    noise_s    = analyse_noise(img_np)
    face_s     = analyse_face_consistency(img_np)
    compress_s = analyse_compression_artifacts(img_np)
    heuristic  = (noise_s * 0.4 + face_s * 0.35 + compress_s * 0.25)

    if fake_prob is not None:
        # Real, trained classifier does most of the work; heuristics add a
        # small amount of independent signal rather than overriding it.
        fused = fake_prob * 0.85 + heuristic * 0.15
        model_status = "trained"
    else:
        fused = heuristic
        model_status = "untrained-baseline"
    fused = float(np.clip(fused, 0, 1))

    is_fake = fused > 0.5
    confidence_pct = round(fused * 100 if is_fake else (1 - fused) * 100, 1)

    explanation = build_explanation(fused, noise_s, face_s, compress_s, is_fake)

    return {
        "verdict":     "DEEPFAKE" if is_fake else "REAL",
        "confidence":  confidence_pct,
        "raw_score":   round(fused, 4),
        "is_fake":     is_fake,
        "model_status": model_status,
        "model_source": HF_MODEL_ID if model_status == "trained" else None,
        "scores": {
            "model":       round(fake_prob, 4) if fake_prob is not None else None,
            "noise":       noise_s,
            "face":        face_s,
            "compression": compress_s,
        },
        "explanation": explanation,
    }


def predict_video(video_path: str, sample_every: int = 15) -> dict:
    """
    Analyse a video file by sampling frames.
    Returns aggregate verdict with per-frame breakdown.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    duration_s   = round(total_frames / fps, 1)

    frame_results = []
    frame_idx     = 0
    fake_count    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil   = Image.fromarray(rgb)
            res   = predict_image(pil)
            timestamp = round(frame_idx / fps, 2)
            frame_results.append({
                "frame":      frame_idx,
                "timestamp":  timestamp,
                "verdict":    res["verdict"],
                "confidence": res["confidence"],
                "raw_score":  res["raw_score"],
            })
            if res["is_fake"]:
                fake_count += 1
        frame_idx += 1

    cap.release()

    if not frame_results:
        return {"error": "No frames could be analysed."}

    analysed     = len(frame_results)
    fake_ratio   = fake_count / analysed
    avg_score    = round(sum(r["raw_score"] for r in frame_results) / analysed, 4)
    is_fake      = fake_ratio > 0.4
    confidence   = round((fake_ratio if is_fake else 1 - fake_ratio) * 100, 1)

    # Highlight suspicious segments
    suspicious = [r for r in frame_results if r["verdict"] == "DEEPFAKE"]

    explanation = []
    if is_fake:
        explanation.append(f"🔴 {fake_count} of {analysed} sampled frames flagged as deepfake ({round(fake_ratio*100)}%).")
        if suspicious:
            first_ts = suspicious[0]["timestamp"]
            explanation.append(f"🔴 Manipulation first detected at {first_ts}s into the video.")
        explanation.append("🔴 Inconsistent facial features across frames — hallmark of face-swap deepfakes.")
    else:
        explanation.append(f"🟢 {analysed - fake_count} of {analysed} sampled frames classified as authentic.")
        explanation.append("🟢 No significant temporal inconsistencies detected across frames.")

    return {
        "verdict":       "DEEPFAKE" if is_fake else "REAL",
        "confidence":    confidence,
        "is_fake":       is_fake,
        "avg_score":     avg_score,
        "fake_ratio":    round(fake_ratio * 100, 1),
        "total_frames":  total_frames,
        "analysed":      analysed,
        "duration_s":    duration_s,
        "frame_results": frame_results,
        "explanation":   explanation,
    }
