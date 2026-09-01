"""
audio_detector.py — Audio deepfake detection
Layer 1: Voice clone detection (MFCC, pitch, jitter, shimmer)
Layer 2: Audio edit/splice detection (noise floor, phase, pitch smoothness)
"""

import numpy as np
import librosa
from scipy.stats import kurtosis, skew
import os


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
    y, sr    = load_audio(filepath)
    duration = float(len(y) / sr)

    features               = extract_voice_features(y, sr)
    clone_score, clone_flags = score_voice_clone(features, duration)
    edit_score, edit_flags, edit_points = detect_audio_edits(y, sr)

    fused   = round(clone_score * 0.55 + edit_score * 0.45, 4)
    is_fake = fused > 0.45
    conf    = round(fused * 100 if is_fake else (1 - fused) * 100, 1)

    return {
        "verdict":     "DEEPFAKE" if is_fake else "REAL",
        "confidence":  conf,
        "is_fake":     is_fake,
        "raw_score":   fused,
        "scores": {
            "voice_clone": clone_score,
            "audio_edit":  edit_score,
            "fused":       fused,
        },
        "explanation": clone_flags + edit_flags,
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
