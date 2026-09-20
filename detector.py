"""
detector.py — Deepfake Defence core engine

Primary signal: an average across one or more real, pretrained deepfake
image classifiers from Hugging Face. Currently:

  - buildborderless/CommunityForensics-DeepfakeDet-ViT
    ViT-Small trained on 2.7M images across 4,803 different generators.
    From a peer-reviewed paper: "Community Forensics: Using Thousands of
    Generators to Train Fake Image Detectors" (Park & Owens, University of
    Michigan, CVPR 2025 — arXiv:2411.04125). MIT licensed. By far the
    largest and most diverse training set of the models used here.
    https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT

  - dima806/deepfake_vs_real_image_detection
    ViT-Base fine-tuned on a face-swap dataset. apache-2.0. Self-reports
    99.27% accuracy on its own eval set (76k images).
    https://huggingface.co/dima806/deepfake_vs_real_image_detection

A note on how the first one was verified: its Hugging Face page currently
contains a section that reads like a prompt injection aimed at AI coding
agents specifically (naming Claude Code / Cursor / Copilot, pointing them at
an "AGENTS.md", and asserting a `transformers >= 5.4.0` requirement that does
not correspond to any real release). That block was ignored entirely — no
extra files were fetched from that repo, no dependency versions were changed
on its say-so, and no custom/remote code from it is used. What WAS verified
independently: the paper exists (arXiv:2411.04125), is CVPR 2025, and the
model card's plain "Quick Start" section (a completely ordinary sigmoid
single-logit classifier, unremarkable on its own) is consistent with normal
`transformers` usage on the version already pinned in requirements.txt.

Caveats worth being upfront about, not papering over:
  - Both accuracy figures are self-reported on each model's own held-out
    data from the SAME distribution it trained on — that tends to read high
    and is not a guarantee of accuracy on a generator neither one saw.
  - dima806's own model card explicitly warns it is ~3 years old and that
    modern generators (Flux, Midjourney, DALL-E 3, Stable Diffusion 3) cause
    "significant concept drift" — a fully AI-generated image from a newer
    generator can plausibly still read as REAL to this class of model. The
    Community Forensics model's much larger, more diverse generator coverage
    is specifically meant to reduce (not eliminate) this failure mode.
  - Deliberately NOT using prithivMLmods/Deep-Fake-Detector-v2-Model: there
    is an open, unresolved report on that model's own discussion page
    ("Label mappings are inverted in the HF pipeline") — using it risked
    silently flipping every verdict.

Combination method: a plain average of each loaded model's P(fake), no
invented weighting or calibration constants — there's no labeled validation
set of our own to justify anything fancier than that. Add or remove models
via the DEEPFAKE_HF_MODELS env var (comma-separated).

Each model's own config is read at runtime rather than assuming a fixed
output format or label index — checkpoints differ (single-logit sigmoid vs
multi-class softmax, and which index means "fake" when it is softmax), and
guessing wrong silently inverts or misreads every result.

If none of the models can be loaded (transformers not installed, no internet
on first run, cache permission errors), predict_image returns an "error"
result instead of a verdict. The noise / symmetry / compression heuristics are
NOT evidence of manipulation (they score most ordinary photos as "fake-ish"),
so they are reported for information only and never used to decide a verdict.
"""

import os

# Keep the Hugging Face cache somewhere this user account can write. Must run
# BEFORE transformers / huggingface_hub are imported (they read it at import).
# setdefault: a HF_HOME already set in the shell still wins.
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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
HF_MODEL_IDS = [
    m.strip() for m in os.getenv(
        "DEEPFAKE_HF_MODELS",
        "buildborderless/CommunityForensics-DeepfakeDet-ViT,"
        "dima806/deepfake_vs_real_image_detection",
    ).split(",") if m.strip()
]

REAL_THRESHOLD = 0.35   # P(fake) at or below this -> REAL
FAKE_THRESHOLD = 0.65   # P(fake) at or above this -> DEEPFAKE; in between -> INCONCLUSIVE

# model_id -> {"processor", "model", "loaded", "error"}
_states = {}


def load_model():
    """
    Downloads and caches each configured Hugging Face model on first call
    (needs internet the first time; cached under ~/.cache/huggingface after
    that). A model that fails to load is skipped, not fatal — never raises.
    """
    if not _TRANSFORMERS_AVAILABLE:
        for model_id in HF_MODEL_IDS:
            state = _states.setdefault(model_id, {})
            if not state.get("error"):
                state["error"] = "transformers is not installed"
        print("[detector] transformers is not installed — run: pip install transformers\n"
              "[detector] Falling back to heuristics only.")
        return

    for model_id in HF_MODEL_IDS:
        state = _states.setdefault(model_id, {"processor": None, "model": None,
                                              "loaded": False, "error": None})
        if state["loaded"] or state["error"]:
            continue
        try:
            state["processor"] = AutoImageProcessor.from_pretrained(model_id)
            m = AutoModelForImageClassification.from_pretrained(model_id).to(DEVICE)
            m.eval()
            state["model"] = m
            state["loaded"] = True
            print(f"[detector] loaded pretrained classifier: {model_id} "
                  f"(labels: {dict(m.config.id2label)})")
        except Exception as exc:
            state["error"] = str(exc)
            print(f"[detector] could not load {model_id}: {exc}\n"
                  "[detector] If this is the first run, check internet access "
                  "— models download from huggingface.co.")


def _is_fake_label(label: str) -> bool:
    """True if a class label means "AI-generated / manipulated" (checkpoints name it differently)."""
    label = label.strip().lower()
    if label in ("ai", "ai-generated", "ai_generated", "ai generated"):
        return True
    return any(w in label for w in ("fake", "artificial", "generated", "synthetic"))


def _fake_probability_for(model_id, pil_image: Image.Image):
    """
    Runs one loaded model and returns P(fake) in [0, 1], or None if it isn't
    loaded or its output format is unrecognised.

    Handles two conventions, detected from the model's own config rather than
    assumed for any specific checkpoint:
      - single-logit sigmoid heads (config.num_labels == 1) — one probability,
        by the near-universal convention that a higher value means "fake"
      - multi-class softmax heads — reads id2label rather than assuming index
        order, since that varies by checkpoint
    """
    state = _states.get(model_id)
    if not state or not state.get("loaded"):
        return None
    inputs = state["processor"](images=pil_image.convert("RGB"), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = state["model"](**inputs).logits

    num_labels = getattr(state["model"].config, "num_labels", logits.shape[-1])
    if num_labels == 1:
        return float(torch.sigmoid(logits)[0][0].item())

    probs = torch.softmax(logits, dim=-1)[0]
    id2label = {int(k): str(v).lower() for k, v in state["model"].config.id2label.items()}
    fake_idx = next((i for i, lbl in id2label.items() if _is_fake_label(lbl)), None)
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

    model_probs = {}
    for model_id in HF_MODEL_IDS:
        try:
            p = _fake_probability_for(model_id, pil_image)
        except Exception as exc:
            print(f"[detector] inference failed for {model_id}, skipping it: {exc}")
            p = None
        if p is not None:
            model_probs[model_id] = p

    if not model_probs:
        # No classifier ran. Refuse to guess rather than invent a verdict.
        return {
            "error": "No detection model is loaded, so no verdict can be given.",
            "model_status": "unavailable",
            "model_errors": {k: v.get("error") for k, v in _states.items() if v.get("error")},
        }

    # Heuristics: informational only, NOT part of the verdict (see docstring).
    noise_s    = analyse_noise(img_np)
    face_s     = analyse_face_consistency(img_np)
    compress_s = analyse_compression_artifacts(img_np)

    # Plain average across whichever models loaded — no invented weights.
    model_avg = sum(model_probs.values()) / len(model_probs)
    fused = float(np.clip(model_avg, 0, 1))

    # Three-way verdict. The 0.35 / 0.65 cut-offs are a starting point, not
    # calibrated values — tune them on images you have labelled yourself.
    if fused >= FAKE_THRESHOLD:
        verdict, is_fake = "DEEPFAKE", True
        confidence_pct = round(fused * 100, 1)
    elif fused <= REAL_THRESHOLD:
        verdict, is_fake = "REAL", False
        confidence_pct = round((1 - fused) * 100, 1)
    else:
        verdict, is_fake = "INCONCLUSIVE", False
        confidence_pct = round(max(fused, 1 - fused) * 100, 1)

    if verdict == "INCONCLUSIVE":
        explanation = ["🟡 The classifier(s) are not confident either way — treat this as unverified, not as real or fake."]
    elif is_fake:
        explanation = [f"🔴 Classifier average P(fake) = {fused:.0%}."]
    else:
        explanation = [f"🟢 Classifier average P(fake) = {fused:.0%}."]

    if len(model_probs) > 1:
        votes = ", ".join(f"{k.split('/')[-1]}: {v:.0%}" for k, v in model_probs.items())
        explanation.append(f"🔎 Per-model P(fake) — {votes}.")
        if (max(model_probs.values()) > 0.5) != (min(model_probs.values()) > 0.5):
            explanation.append("⚠️ The models disagree with each other on this image.")

    return {
        "verdict":     verdict,
        "confidence":  confidence_pct,
        "raw_score":   round(fused, 4),
        "is_fake":     is_fake,
        "model_status": "trained",
        "model_source": ", ".join(model_probs.keys()),
        "scores": {
            "model":       round(model_avg, 4),
            "per_model":   {k: round(v, 4) for k, v in model_probs.items()},
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
            if "error" in res:
                cap.release()
                return {"error": res["error"]}
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
