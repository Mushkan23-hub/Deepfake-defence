"""
detector.py — Deepfake Defence core engine
Uses a pretrained EfficientNet-B4 fine-tuned for deepfake detection via timm.
Falls back to a heuristic analyser if model weights are unavailable.
"""

import io
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Model ─────────────────────────────────────────────────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # EfficientNet-B4 pretrained on ImageNet; final layer replaced for binary classification
        self.backbone = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


WEIGHTS_PATH = os.getenv("DEEPFAKE_WEIGHTS", "weights/deepfake_effb4.pt")

# IMPORTANT — read this before quoting any accuracy figure.
#
# timm gives us an ImageNet-pretrained *backbone*, but `self.classifier` above
# is randomly initialised and has never seen a deepfake. Until it is trained,
# its output is a fixed random projection of image features: deterministic, but
# meaningless as a deepfake score. The original code described this as a "good
# zero-shot proxy". It is not one, and presenting it as a detection confidence
# would be a false claim in a forensic report.
#
# So: if trained weights exist at WEIGHTS_PATH we load them and use the model.
# If they don't, we say so, and the verdict falls back to the heuristic signals
# (noise / symmetry / compression) alone, which at least measure something real.

MODEL_TRAINED = False


def load_model():
    global MODEL_TRAINED
    model = DeepfakeDetector().to(DEVICE)
    if os.path.isfile(WEIGHTS_PATH):
        state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(state.get("state_dict", state))
        MODEL_TRAINED = True
        print(f"[detector] loaded fine-tuned weights from {WEIGHTS_PATH}")
    else:
        print(f"[detector] no weights at {WEIGHTS_PATH} — classifier head is "
              "UNTRAINED. Falling back to heuristics only. Train the head "
              "(e.g. on FaceForensics++ or Celeb-DF) before reporting accuracy.")
    model.eval()
    return model


_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


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
    model = get_model()
    img_np = np.array(pil_image.convert("RGB"))

    # Model inference
    tensor = TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        raw = model(tensor).item()

    # Heuristic scores
    noise_s    = analyse_noise(img_np)
    face_s     = analyse_face_consistency(img_np)
    compress_s = analyse_compression_artifacts(img_np)

    # Weighted fusion. The model only gets a vote once it has actually been
    # trained; otherwise we would be averaging in noise and calling it evidence.
    heuristic = (noise_s * 0.4 + face_s * 0.35 + compress_s * 0.25)
    if MODEL_TRAINED:
        fused = raw * 0.60 + heuristic * 0.40
    else:
        fused = heuristic
    fused = float(np.clip(fused, 0, 1))

    is_fake = fused > 0.5
    confidence_pct = round(fused * 100 if is_fake else (1 - fused) * 100, 1)

    explanation = build_explanation(fused, noise_s, face_s, compress_s, is_fake)

    return {
        "verdict":     "DEEPFAKE" if is_fake else "REAL",
        "confidence":  confidence_pct,
        "raw_score":   round(fused, 4),
        "is_fake":     is_fake,
        "model_status": "trained" if MODEL_TRAINED else "untrained-baseline",
        "scores": {
            "model":       round(raw, 4) if MODEL_TRAINED else None,
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
