"""
forensics.py — ExifTool, ELA, ffprobe
"""
import subprocess, json, os, tempfile
import numpy as np
from PIL import Image
import cv2


def run_exiftool(filepath):
    try:
        result = subprocess.run(
            ["exiftool", "-j", "-a", "-u", filepath],
            capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {"error": result.stderr.strip()}
        data = json.loads(result.stdout)
        if not data:
            return {"error": "No metadata found."}
        meta = data[0]
        relevant = {
            "File Type":         meta.get("FileType", "Unknown"),
            "MIME Type":         meta.get("MIMEType", "Unknown"),
            "Image Size":        meta.get("ImageSize") or meta.get("ImageWidth", "?"),
            "Camera Make":       meta.get("Make", "Not found"),
            "Camera Model":      meta.get("Model", "Not found"),
            "Software":          meta.get("Software", "Not found"),
            "Date Created":      meta.get("DateTimeOriginal") or meta.get("CreateDate", "Not found"),
            "Date Modified":     meta.get("FileModifyDate", "Not found"),
            "GPS Coordinates":   meta.get("GPSPosition", "Not found"),
            "Color Space":       meta.get("ColorSpace", "Not found"),
            "Bit Depth":         meta.get("BitDepth") or meta.get("BitsPerSample", "Not found"),
            "Encoding Software": meta.get("EncoderSettings") or meta.get("Encoder", "Not found"),
        }
        flags = []
        if relevant["Camera Make"] == "Not found" and relevant["Camera Model"] == "Not found":
            flags.append("⚠️ No camera metadata — file may be AI-generated or screenshot.")
        if relevant["Software"] not in ["Not found", "Unknown"]:
            flags.append(f"⚠️ Edited with software: {relevant['Software']}")
        if relevant["Date Created"] == "Not found":
            flags.append("⚠️ No creation timestamp — metadata may have been stripped.")
        if not flags:
            flags.append("✅ Metadata looks normal.")
        return {"metadata": relevant, "flags": flags, "raw_count": len(meta)}
    except FileNotFoundError:
        return {"error": "ExifTool not installed. Install and add to PATH."}
    except Exception as e:
        return {"error": str(e)}


def run_ela(filepath, quality=95):
    try:
        original = Image.open(filepath).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        original.save(tmp_path, "JPEG", quality=quality)
        resaved  = Image.open(tmp_path).convert("RGB")
        os.unlink(tmp_path)

        orig_np   = np.array(original).astype(np.float32)
        resave_np = np.array(resaved).astype(np.float32)
        ela_np    = np.abs(orig_np - resave_np)
        ela_scaled = np.clip(ela_np * 10, 0, 255).astype(np.uint8)

        mean_error = float(ela_np.mean())
        max_error  = float(ela_np.max())
        std_error  = float(ela_np.std())
        suspicious = mean_error > 8.0 or std_error > 15.0

        gray_ela = cv2.cvtColor(ela_scaled, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray_ela, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hotspot = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            iH, iW = orig_np.shape[:2]
            hotspot = {"x_pct": round(x/iW*100,1), "y_pct": round(y/iH*100,1),
                       "w_pct": round(w/iW*100,1), "h_pct": round(h/iH*100,1)}

        flags = []
        if mean_error > 8.0:
            flags.append(f"⚠️ High average error level ({mean_error:.2f}) — image may be composited.")
        if std_error > 15.0:
            flags.append(f"⚠️ Uneven error distribution (σ={std_error:.2f}) — specific region likely edited.")
        if hotspot:
            flags.append(f"⚠️ Highest manipulation region at x={hotspot['x_pct']}%, y={hotspot['y_pct']}%")
        if not flags:
            flags.append("✅ Error levels consistent — no obvious editing detected.")

        return {"mean_error": round(mean_error,3), "max_error": round(max_error,3),
                "std_error": round(std_error,3), "suspicious": suspicious,
                "hotspot": hotspot, "flags": flags}
    except Exception as e:
        return {"error": str(e)}


def run_ffprobe(filepath):
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", filepath
        ], capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return {"error": result.stderr.strip()}
        data    = json.loads(result.stdout)
        fmt     = data.get("format", {})
        streams = data.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

        video_info = {}
        if video_streams:
            v = video_streams[0]
            fps_raw = v.get("r_frame_rate", "0/1")
            try:
                num, den = fps_raw.split("/"); fps = round(int(num)/int(den), 2)
            except Exception:
                fps = fps_raw
            video_info = {
                "Codec":       v.get("codec_name","Unknown"),
                "Resolution":  f"{v.get('width','?')}x{v.get('height','?')}",
                "Frame Rate":  f"{fps} fps",
                "Bit Rate":    f"{int(v.get('bit_rate',0))//1000} kbps" if v.get("bit_rate") else "Unknown",
                "Pixel Format":v.get("pix_fmt","Unknown"),
            }

        audio_info = {}
        if audio_streams:
            a = audio_streams[0]
            audio_info = {
                "Codec":      a.get("codec_name","Unknown"),
                "Sample Rate":f"{a.get('sample_rate','?')} Hz",
                "Channels":   a.get("channels","?"),
            }

        fmt_tags = fmt.get("tags", {})
        format_info = {
            "Container":        fmt.get("format_long_name","Unknown"),
            "Duration":         f"{float(fmt.get('duration',0)):.2f}s" if fmt.get("duration") else "Unknown",
            "File Size":        f"{int(fmt.get('size',0))//1024} KB" if fmt.get("size") else "Unknown",
            "Encoding Software":fmt_tags.get("encoder") or fmt_tags.get("Encoder","Not found"),
            "Creation Time":    fmt_tags.get("creation_time","Not found"),
        }

        flags = []
        enc = format_info["Encoding Software"].lower()
        if "lavf" in enc or "ffmpeg" in enc:
            flags.append("⚠️ Re-encoded with FFmpeg — possible deepfake pipeline post-processing.")
        if not audio_streams:
            flags.append("⚠️ No audio stream — unusual for authentic recorded video.")
        if not flags:
            flags.append("✅ No obvious forensic anomalies in video stream.")

        return {"format": format_info, "video": video_info, "audio": audio_info, "flags": flags}
    except FileNotFoundError:
        return {"error": "ffprobe not installed. Install ffmpeg and add to PATH."}
    except Exception as e:
        return {"error": str(e)}
