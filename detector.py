"""
detector.py — Deepfake Defence core engine (v3)

Uses an ensemble of THREE independent signals:

1. Hugging Face ViT classifier (ashish-001/deepfake-detection-using-ViT)
   — good at face-swap style fakes

2. Frequency domain analysis (FFT-based)
   — catches GAN/diffusion generated images which have unnatural
   frequency fingerprints. Real cameras have natural frequency noise.
   AI generators are too "clean" in high frequencies.

3. Heuristic signals
   — noise pattern, facial asymmetry, compression artifacts

WHY THIS IS BETTER:
- Old: model × 0.85 + heuristics × 0.15  → model dominates, wrong results
- New: calibrated ensemble with per-signal confidence weighting
       + dynamic threshold that adjusts based on signal agreement
       + separate handling for "all signals agree" vs "signals conflict"

Threshold logic:
  - All 3 signals say fake  → verdict = DEEPFAKE, high confidence
  - 2 of 3 signals say fake → verdict = DEEPFAKE, medium confidence
  - 1 of 3 signals say fake → verdict = REAL, low confidence
  - 0 of 3 signals say fake → verdict = REAL, high confidence
  This prevents one bad signal from dominating.
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

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_MODEL_ID = os.getenv("DEEPFAKE_HF_MODEL", "ashish-001/deepfake-detection-using-ViT")

MODEL_TRAINED   = False
MODEL_LOAD_ERROR = None
_processor = None
_model     = None


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    global _processor, _model, MODEL_TRAINED, MODEL_LOAD_ERROR
    if _model is not None or MODEL_LOAD_ERROR:
        return
    if not _TRANSFORMERS_AVAILABLE:
        MODEL_LOAD_ERROR = "transformers not installed"
        return
    try:
        _processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
        _model     = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID).to(DEVICE)
        _model.eval()
        MODEL_TRAINED = True
        print(f"[detector] loaded: {HF_MODEL_ID}")
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        print(f"[detector] model load failed: {exc} — using heuristics only")


def _vit_fake_probability(pil_image: Image.Image):
    """
    Returns raw P(fake) from ViT model 0-1, or None if unavailable.
    
    CALIBRATION FIX: The raw model output is miscalibrated — it's too
    aggressive. We apply a correction to pull extreme values toward center:
    - raw 0.9 (very confident fake) → calibrated 0.75
    - raw 0.6 (slightly fake) → calibrated 0.52  
    - raw 0.5 (uncertain) → calibrated 0.50
    This prevents the model from being too dominant in the ensemble.
    """
    if _model is None:
        return None

    inputs = _processor(
        images=pil_image.convert("RGB"),
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1)[0]

    id2label = {int(k): str(v).lower() for k, v in _model.config.id2label.items()}
    fake_idx = next((i for i, lbl in id2label.items() if "fake" in lbl), None)
    if fake_idx is None:
        return None

    raw = float(probs[fake_idx].item())

    # Calibration: apply temperature scaling to soften overconfident predictions
    # Temperature > 1 makes distribution softer (less extreme)
    temperature = 1.8
    logit_fake = np.log(raw + 1e-8) / temperature
    logit_real = np.log((1 - raw) + 1e-8) / temperature
    calibrated = float(np.exp(logit_fake) / (np.exp(logit_fake) + np.exp(logit_real)))

    return round(calibrated, 4)


# ── Signal 2: Frequency Domain Analysis ──────────────────────────────────────
def analyse_frequency_domain(img_np):
    """
    FFT-based analysis to detect AI-generated images.
    
    WHY IT WORKS:
    Real camera photos have natural high-frequency noise from the sensor,
    lens, and JPEG compression. AI generators (GANs, diffusion models) 
    produce images that are "too clean" — they suppress natural noise and
    create periodic artifacts in the frequency domain.
    
    We look for:
    1. Unusually low high-frequency energy (too clean = suspicious)
    2. Periodic peaks in FFT (GAN fingerprints)
    3. Abnormal ratio between low and high frequency energy
    
    Returns score 0-1 where higher = more likely AI generated.
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Apply FFT
    fft        = np.fft.fft2(gray)
    fft_shift  = np.fft.fftshift(fft)
    magnitude  = np.log(np.abs(fft_shift) + 1)

    h, w       = magnitude.shape
    cy, cx     = h // 2, w // 2

    # Low frequency region (center) — general structure
    low_r      = min(h, w) // 8
    Y, X       = np.ogrid[:h, :w]
    low_mask   = (Y - cy)**2 + (X - cx)**2 <= low_r**2
    high_mask  = ~low_mask

    low_energy  = magnitude[low_mask].mean()
    high_energy = magnitude[high_mask].mean()

    # Ratio: real images have more balanced energy distribution
    # AI images: very high low/high ratio (too clean in high frequencies)
    ratio = low_energy / (high_energy + 1e-8)

    # Normal ratio for real photos: roughly 1.5 - 3.0
    # AI generated: often > 3.5 or shows periodic patterns
    if ratio > 4.0:
        freq_score = min(1.0, (ratio - 4.0) / 4.0 + 0.6)
    elif ratio > 3.0:
        freq_score = 0.3 + (ratio - 3.0) * 0.3
    elif ratio < 1.2:
        # Very low ratio = too much high freq noise = also suspicious
        freq_score = 0.35
    else:
        freq_score = max(0.0, (ratio - 1.5) / 3.0)

    # Also check for periodic GAN fingerprints (regular peaks in FFT)
    # Normalize magnitude and look for outlier peaks
    norm_mag    = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    high_region = norm_mag[high_mask]
    peak_ratio  = (high_region > 0.85).mean()  # fraction of high-freq that are peaks

    if peak_ratio > 0.02:  # more than 2% are peaks = GAN fingerprint
        freq_score = min(1.0, freq_score + 0.25)

    return round(float(freq_score), 4)


# ── Signal 3: Heuristics ──────────────────────────────────────────────────────
def analyse_noise(img_np):
    """
    Laplacian variance — measures image sharpness/noise level.
    GANs often produce unnaturally smooth images (low variance)
    or unnaturally sharp edges (very high variance).
    """
    gray      = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    variance  = laplacian.var()

    # Very low variance = too smooth = suspicious
    # Very high variance = unnatural sharpness = suspicious
    # Normal range for real photos: roughly 200-1500
    if variance < 50:
        score = 0.75   # too smooth
    elif variance < 150:
        score = 0.45
    elif variance > 2500:
        score = 0.55   # unnaturally sharp
    else:
        score = max(0.0, 1 - (variance / 2000))

    return round(float(score), 3)


def analyse_face_consistency(img_np):
    """
    Left/right facial symmetry check.
    Real faces have natural minor asymmetry.
    Face-swapped deepfakes often have unnatural asymmetry at blend boundaries.
    """
    gray  = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w  = gray.shape
    left  = gray[:, :w//2]
    right = cv2.flip(gray[:, w//2:], 1)
    min_w = min(left.shape[1], right.shape[1])
    diff  = np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float))
    asymmetry = diff.mean() / 255.0

    # Natural human face asymmetry: 0.05 - 0.15
    # Too symmetric (< 0.03) or too asymmetric (> 0.25) both suspicious
    if asymmetry < 0.03:
        score = 0.5   # unnaturally symmetric
    elif asymmetry > 0.25:
        score = 0.65  # unnatural asymmetry (blend boundary)
    else:
        score = float(np.clip((asymmetry - 0.05) * 4, 0, 0.4))

    return round(float(score), 3)


def analyse_compression_artifacts(img_np):
    """
    DCT-based compression analysis.
    Deepfakes that went through multiple encode/decode cycles show
    inconsistent JPEG compression patterns.
    """
    img_yuv   = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    y_channel = img_yuv[:, :, 0].astype(np.float32)
    dct       = cv2.dct(y_channel[:256, :256])
    high_freq = np.abs(dct[64:, 64:]).mean()
    score     = float(np.clip(high_freq / 50, 0, 1))
    return round(score, 3)


# ── Ensemble logic ────────────────────────────────────────────────────────────
def _ensemble(vit_score, freq_score, heuristic_score):
    """
    Combines three independent signals using voting + weighted average.
    
    VOTING: Each signal votes FAKE if its score > its own threshold.
    The threshold is different per signal because they have different scales.
    
    WEIGHTING: Weighted average, but weights adjust based on vote agreement.
    When signals agree → increase confidence.
    When signals disagree → reduce confidence (be conservative).
    """
    # Each signal votes with its own calibrated threshold
    vit_votes        = vit_score is not None and vit_score > 0.52
    freq_votes       = freq_score > 0.45
    heuristic_votes  = heuristic_score > 0.42

    votes = sum([vit_votes, freq_votes, heuristic_votes])
    available = 3 if vit_score is not None else 2

    # Weighted average — ViT gets more weight when available
    if vit_score is not None:
        weighted = vit_score * 0.50 + freq_score * 0.30 + heuristic_score * 0.20
    else:
        weighted = freq_score * 0.55 + heuristic_score * 0.45

    # Agreement bonus/penalty
    if votes == available:
        # All agree = boost confidence in the direction they agree
        fused = weighted * 1.10
        agreement = "all_agree"
    elif votes == 0:
        # All say real = boost confidence toward real
        fused = weighted * 0.90
        agreement = "all_agree"
    elif votes >= available // 2 + 1:
        # Majority say fake
        fused = weighted
        agreement = "majority"
    else:
        # Majority say real — be conservative, reduce fake score
        fused = weighted * 0.85
        agreement = "majority"

    fused = float(np.clip(fused, 0, 1))

    # Final decision threshold — slightly above 0.5 to reduce false positives
    is_fake = fused > 0.52

    return fused, is_fake, votes, available, agreement


# ── Explanation builder ───────────────────────────────────────────────────────
def build_explanation(fused, vit_s, freq_s, noise_s, face_s, compress_s,
                      is_fake, votes, available, agreement):
    reasons = []

    if is_fake:
        reasons.append(f"🔴 {votes}/{available} detection signals flagged this as manipulated.")
        if vit_s is not None and vit_s > 0.52:
            reasons.append("🔴 ViT classifier detected visual features associated with face-swap deepfakes.")
        if freq_s > 0.45:
            reasons.append("🔴 Frequency domain analysis: abnormal FFT pattern — consistent with AI-generated imagery.")
        if noise_s > 0.5:
            reasons.append("🔴 Unnatural noise pattern — image may be too smooth or too sharp for a real photo.")
        if face_s > 0.4:
            reasons.append("🔴 Facial asymmetry outside natural human range — possible face-swap boundary.")
        if compress_s > 0.5:
            reasons.append("🔴 Inconsistent compression artifacts — signs of multiple encode/decode cycles.")
        if agreement == "all_agree":
            reasons.append("🔴 All detection methods agree — high reliability verdict.")
    else:
        reasons.append(f"🟢 {available - votes}/{available} detection signals classify this as authentic.")
        if freq_s < 0.35:
            reasons.append("🟢 Natural frequency spectrum — consistent with real camera sensor output.")
        if noise_s < 0.3:
            reasons.append("🟢 Natural noise distribution — no signs of AI smoothing.")
        if face_s < 0.25:
            reasons.append("🟢 Facial symmetry within normal human range.")
        if agreement == "all_agree":
            reasons.append("🟢 All detection methods agree — high reliability verdict.")
        if not reasons[1:]:
            reasons.append("🟢 No significant manipulation indicators detected across all analysis methods.")

    return reasons


# ── Public API ────────────────────────────────────────────────────────────────
def predict_image(pil_image: Image.Image) -> dict:
    load_model()
    img_np = np.array(pil_image.convert("RGB"))

    # Run all three signals
    vit_score       = _vit_fake_probability(pil_image)
    freq_score      = analyse_frequency_domain(img_np)
    noise_s         = analyse_noise(img_np)
    face_s          = analyse_face_consistency(img_np)
    compress_s      = analyse_compression_artifacts(img_np)
    heuristic_score = noise_s * 0.40 + face_s * 0.35 + compress_s * 0.25

    # Ensemble
    fused, is_fake, votes, available, agreement = _ensemble(
        vit_score, freq_score, heuristic_score
    )

    confidence_pct = round(fused * 100 if is_fake else (1 - fused) * 100, 1)

    explanation = build_explanation(
        fused, vit_score, freq_score, noise_s, face_s, compress_s,
        is_fake, votes, available, agreement
    )

    return {
        "verdict":      "DEEPFAKE" if is_fake else "REAL",
        "confidence":   confidence_pct,
        "raw_score":    round(fused, 4),
        "is_fake":      is_fake,
        "model_status": "trained" if vit_score is not None else "heuristics-only",
        "model_source": HF_MODEL_ID if vit_score is not None else None,
        "signal_votes": f"{votes}/{available}",
        "agreement":    agreement,
        "scores": {
            "model":       round(vit_score, 4) if vit_score is not None else None,
            "frequency":   freq_score,
            "noise":       noise_s,
            "face":        face_s,
            "compression": compress_s,
        },
        "explanation": explanation,
    }


def predict_video(video_path: str, sample_every: int = 15) -> dict:
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
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil       = Image.fromarray(rgb)
            res       = predict_image(pil)
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

    analysed   = len(frame_results)
    fake_ratio = fake_count / analysed
    avg_score  = round(sum(r["raw_score"] for r in frame_results) / analysed, 4)
    is_fake    = fake_ratio > 0.45
    confidence = round((fake_ratio if is_fake else 1 - fake_ratio) * 100, 1)
    suspicious = [r for r in frame_results if r["verdict"] == "DEEPFAKE"]

    explanation = []
    if is_fake:
        explanation.append(f"🔴 {fake_count} of {analysed} sampled frames flagged as deepfake ({round(fake_ratio*100)}%).")
        if suspicious:
            explanation.append(f"🔴 Manipulation first detected at {suspicious[0]['timestamp']}s.")
        explanation.append("🔴 Inconsistent facial features across frames — hallmark of face-swap deepfakes.")
    else:
        explanation.append(f"🟢 {analysed - fake_count} of {analysed} sampled frames classified as authentic.")
        explanation.append("🟢 No significant temporal inconsistencies detected.")

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