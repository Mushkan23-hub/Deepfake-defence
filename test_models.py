"""
test_models.py — run the detector over a folder of REAL images and a folder of
AI-generated images and print each model's P(fake) side by side.

Usage (from the project folder, venv active):
    python test_models.py "C:/path/to/real_folder" "C:/path/to/ai_folder"

Reading the output: a working model gives LOW numbers on the REAL rows and HIGH
numbers on the AI rows. If a model gives ~0.00 on everything, it can't see that
kind of image. If it's high on real and low on AI, its direction may be flipped.
"""
import glob
import os
import sys

from PIL import Image

import detector

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def run(folder, tag):
    files = sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(EXTS))
    if not files:
        print(f"(no images found in {folder})")
    for path in files:
        try:
            img = Image.open(path)
        except Exception as exc:
            print(f"{tag:5} {os.path.basename(path)[:28]:28} cannot open: {exc}")
            continue
        r = detector.predict_image(img)
        if "error" in r:
            print("ERROR:", r["error"])
            for k, v in r.get("model_errors", {}).items():
                print("  ", k, "->", v)
            sys.exit(1)
        per = "  ".join(f"{k.split('/')[-1][:22]}={v:.2f}" for k, v in r["scores"]["per_model"].items())
        print(f"{tag:5} {os.path.basename(path)[:28]:28} {r['verdict']:12} avg={r['raw_score']:.2f}  {per}")


if len(sys.argv) != 3:
    sys.exit(__doc__)

run(sys.argv[1], "REAL")
run(sys.argv[2], "AI")
