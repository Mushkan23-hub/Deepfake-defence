"""
audio_detector.py — Audio deepfake detection

Primary signal: a real, pretrained voice-spoof classifier
(MelodyMachine/Deepfake-audio-detection-V2 on Hugging Face — wav2vec2-base
fine-tuned for real-vs-synthetic speech, apache-2.0, self-reports 99.7% eval
accuracy: https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2).

Worth being skeptical of that number: it's measured on held-out data from the
*same* training distribution, which tends to run high and doesn't guarantee
it generalises to voice-cloning tools it never saw in training. Treat it as
"a real trained model, meaningfully better than hand-tuned thresholds" rather
than "99.7% accurate on anything you throw at it".

Secondary signal (used always, and as the sole signal if the model can't be
loaded): hand-built heuristics on MFCC variance, jitter, spectral flatness
etc. These are unvalidated proxies, not a trained detector — see
`model_status` in the returned dict.

Layer 1: Voice clone detection (MFCC, pitch, jitter, shimmer) — heuristic
Layer 2: Audio edit/splice detection (noise floor, phase, pitch smoothness) — heuristic
"""

import os
import numpy as np
import librosa
from scipy.stats import kurtosis, skew
import torch

try:
    from transformers import AutoProcessor, AutoModelForAudioClassification
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_AUDIO_MODEL_ID = os.getenv("DEEPFAKE_AUDIO_HF_MODEL", "MelodyMachine/Deepfake-audio-detection-V2")

MODEL_TRAINED = False
MODEL_LOAD_ERROR = None
_processor = None
_model = None


def load_audio_model():
    """Downloads and caches the HF model on first call. Never raises."""
    global _processor, _model, MODEL_TRAINED, MODEL_LOAD_ERROR

    if _model is not None or MODEL_LOAD_ERROR:
        return

    if not _TRANSFORMERS_AVAILABLE:
        MODEL_LOAD_ERROR = "transformers is not installed"
        print(f"[audio_detector] {MODEL_LOAD_ERROR} — run: pip install transformers\n"
              "[audio_detector] Falling back to heuristics only.")
        return

    try:
        _processor = AutoProcessor.from_pretrained(HF_AUDIO_MODEL_ID)
        _model = AutoModelForAudioClassification.from_pretrained(HF_AUDIO_MODEL_ID).to(DEVICE)
        _model.eval()
        MODEL_TRAINED = True
        print(f"[audio_detector] loaded pretrained classifier: {HF_AUDIO_MODEL_ID}")
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        print(f"[audio_detector] could not load {HF_AUDIO_MODEL_ID}: {exc}\n"
              "[audio_detector] Falling back to heuristics only. If this is "
              "the first run, check internet access — the model downloads "
              "from huggingface.co.")


def _fake_probability(filepath):
    """
    Runs the HF model at its expected 16kHz mono sample rate. Reads the
    model's own id2label mapping rather than assuming index order.
    """
    if _model is None:
        return None
    y, _ = librosa.load(filepath, sr=16000, mono=True)
    inputs = _processor(y, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    id2label = {int(k): str(v).lower() for k, v in _model.config.id2label.items()}
    fake_idx = next((i for i, lbl in id2label.items() if "fake" in lbl or "spoof" in lbl), None)
    if fake_idx is None:
        return None
    return float(probs[fake_idx].item())


def load_audio(filepath, sr=22050):
    y, sr_out = librosa.load(filepath, sr=sr, mono=True)
    return y, sr_out


def extract_voice_features(y, sr):
    mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_var  = float(np.var(mfccs, axis=1).mean())
    mfcc_mean = float(np.mean(np.abs(mfccs)))

    zcr      = librosa.feature.zero_crossing_rate(y)[0]
    zcr_var  = float(np.var(zcr))
    zcr_mean = float(np.mean(zcr))

    spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr)
        f0_clean   = f0[~np.isnan(f0)]
        pitch_var  = float(np.var(f0_clean))  if len(f0_clean) > 0 else 0.0
        pitch_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
        jitter     = float(np.mean(np.abs(np.diff(f0_clean))) / (pitch_mean + 1e-8)) if len(f0_clean) > 1 else 0.0
        shimmer    = float(np.std(f0_clean) / (pitch_mean + 1e-8)) if len(f0_clean) > 1 else 0.0
    except Exception:
        pitch_var = pitch_mean = jitter = shimmer = 0.0

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db   = librosa.power_to_db(mel_spec, ref=np.max)
    mel_kurt = float(kurtosis(mel_db.flatten()))
    mel_skew = float(skew(mel_db.flatten()))

    return {
        "mfcc_variance":      round(mfcc_var, 4),
        "mfcc_mean":          round(mfcc_mean, 4),
        "zcr_variance":       round(zcr_var, 6),
        "zcr_mean":           round(zcr_mean, 4),
        "spectral_centroid":  round(float(np.mean(spec_centroid)), 2),
        "spectral_flatness":  round(float(np.mean(spec_flatness)), 6),
        "pitch_mean_hz":      round(pitch_mean, 2),
        "pitch_variance":     round(pitch_var, 2),
        "jitter":             round(jitter, 6),
        "shimmer":            round(shimmer, 6),
        "mel_kurtosis":       round(mel_kurt, 4),
        "mel_skew":           round(mel_skew, 4),
    }


def score_voice_clone(features, duration):
    score = 0.0
    flags = []
    if features["mfcc_variance"] < 30:
        score += 0.25
        flags.append("🔴 MFCC variance unusually low — voice lacks natural variation.")
    if features["zcr_variance"] < 0.0005:
        score += 0.20
        flags.append("🔴 Zero-crossing rate too regular — may indicate synthesised voice.")
    if features["jitter"] < 0.001 and features["pitch_mean_hz"] > 0:
        score += 0.20
        flags.append("🔴 Pitch jitter near zero — real voices always have micro-variation.")
    if features["mel_kurtosis"] > 5.0:
        score += 0.20
        flags.append("🔴 Mel spectrogram kurtosis high — distribution too regular for natural speech.")
    if features["spectral_flatness"] > 0.1:
        score += 0.15
        flags.append("🔴 High spectral flatness — unusual noise profile for human voice.")
    if duration < 3.0 and features["pitch_variance"] < 10:
        score += 0.10
        flags.append("⚠️ Very short clip with unnaturally stable pitch.")
    score = min(score, 1.0)
    if not flags:
        flags.append("🟢 Voice features consistent with authentic human speech.")
    return round(score, 4), flags


def detect_audio_edits(y, sr):
    score = 0.0
    flags = []
    edit_points = []

    # Noise floor breaks
    rms        = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_db     = librosa.amplitude_to_db(rms, ref=np.max)
    rms_diff   = np.abs(np.diff(rms_db))
    splice_idx = np.where(rms_diff > 40)[0]
    if len(splice_idx) > 0:
        score += min(0.30, len(splice_idx) * 0.10)
        timestamps = [round(t * 512 / sr, 2) for t in splice_idx]
        edit_points.extend(timestamps)
        flags.append(f"🔴 {len(splice_idx)} sudden energy discontinuities — possible splice points at: {timestamps[:5]}s")

    # Phase discontinuity
    stft        = librosa.stft(y, n_fft=2048, hop_length=512)
    phase       = np.angle(stft)
    phase_diff  = np.diff(phase, axis=1)
    phase_jumps = np.mean(np.abs(phase_diff) > 2.5, axis=0)
    big_phase   = np.sum(phase_jumps > 0.3)
    if big_phase > 2:
        score += 0.25
        flags.append(f"🔴 Phase discontinuities at {big_phase} frames — possible audio splicing.")

    # Pitch contour smoothness
    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 10:
            smoothness = float(np.mean(np.abs(np.diff(f0_clean))))
            if smoothness < 2.0:
                score += 0.20
                flags.append(f"🔴 Pitch contour unnaturally smooth (Δ={smoothness:.2f}Hz) — possible pitch-shifting.")
    except Exception:
        pass

    # Background noise inconsistency
    seg_len    = sr * 2
    n_segments = len(y) // seg_len
    if n_segments >= 3:
        noise_levels = []
        for i in range(n_segments):
            seg = y[i*seg_len:(i+1)*seg_len]
            seg_rms = librosa.feature.rms(y=seg)[0]
            noise_levels.append(float(np.percentile(seg_rms, 10)))
        noise_var = np.std(noise_levels) / (np.mean(noise_levels) + 1e-8)
        if noise_var > 0.5:
            score += 0.25
            flags.append(f"🔴 Background noise inconsistent across segments (σ={noise_var:.2f}) — possible editing.")

    score = min(score, 1.0)
    if not flags:
        flags.append("🟢 No significant edit or splice indicators detected.")
    return round(score, 4), flags, edit_points


def predict_audio(filepath: str) -> dict:
    load_audio_model()

    y, sr    = load_audio(filepath)
    duration = float(len(y) / sr)

    features                  = extract_voice_features(y, sr)
    clone_score, clone_flags  = score_voice_clone(features, duration)
    edit_score, edit_flags, edit_points = detect_audio_edits(y, sr)
    heuristic = round(clone_score * 0.55 + edit_score * 0.45, 4)

    fake_prob = None
    try:
        fake_prob = _fake_probability(filepath)
    except Exception as exc:
        print(f"[audio_detector] model inference failed, using heuristics only: {exc}")

    if fake_prob is not None:
        # Trained classifier does most of the work; heuristics add a smaller
        # amount of independent, explainable signal alongside it.
        fused = fake_prob * 0.85 + heuristic * 0.15
        model_status = "trained"
        lead_note = ["🔎 Verdict is led by a trained voice-spoof classifier; "
                     "the signals below are supporting heuristic detail."]
    else:
        fused = heuristic
        model_status = "untrained-baseline"
        lead_note = []
    fused   = round(float(np.clip(fused, 0, 1)), 4)
    is_fake = fused > 0.5
    conf    = round(fused * 100 if is_fake else (1 - fused) * 100, 1)

    return {
        "verdict":      "DEEPFAKE" if is_fake else "REAL",
        "confidence":   conf,
        "is_fake":      is_fake,
        "raw_score":    fused,
        "model_status": model_status,
        "model_source": HF_AUDIO_MODEL_ID if model_status == "trained" else None,
        "scores": {
            "model":       round(fake_prob, 4) if fake_prob is not None else None,
            "voice_clone": clone_score,
            "audio_edit":  edit_score,
        },
        "explanation": lead_note + clone_flags + edit_flags,
        "forensics": {
            "audio": {
                "duration_seconds": round(duration, 2),
                "sample_rate":      sr,
                "clone_score":      round(clone_score * 100, 1),
                "edit_score":       round(edit_score  * 100, 1),
                "edit_points_sec":  edit_points[:10],
                "voice_features":   features,
            }
        }
    }
