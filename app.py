import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

APP_VERSION = "1.0.0"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def analyze_video(path: str, sample_every_seconds: float, diff_threshold: float,
                  black_threshold: int, max_frames: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("OpenCV could not open this video. Try a common MP4/MOV/AVI file.")

    fps = safe_float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    frame_count_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (frame_count_reported / fps) if fps > 0 and frame_count_reported > 0 else None

    frame_rows = []
    sampled_images = []
    previous_gray = None
    frame_index = 0
    sampled_count = 0
    black_count = 0
    duplicate_like_count = 0
    change_scores = []
    sample_step = max(1, int(round((fps if fps > 0 else 1) * sample_every_seconds)))
    max_scan_frames = max_frames if max_frames > 0 else None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        is_black = mean_brightness <= black_threshold
        if is_black:
            black_count += 1

        # Mean absolute difference between consecutive decoded frames.
        diff_score = None
        if previous_gray is not None:
            if previous_gray.shape != gray.shape:
                gray_for_diff = cv2.resize(gray, (previous_gray.shape[1], previous_gray.shape[0]))
            else:
                gray_for_diff = gray
            diff_score = float(np.mean(cv2.absdiff(previous_gray, gray_for_diff)))
            change_scores.append(diff_score)
            if diff_score <= 0.25:
                duplicate_like_count += 1
        previous_gray = gray

        if frame_index % sample_step == 0:
            timestamp = (frame_index / fps) if fps > 0 else None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            thumb = Image.fromarray(rgb)
            thumb.thumbnail((480, 270))
            sampled_images.append((frame_index, timestamp, thumb))
            sampled_count += 1

        frame_rows.append({
            "frame_index": frame_index,
            "timestamp_seconds": round(frame_index / fps, 4) if fps > 0 else None,
            "mean_brightness": round(mean_brightness, 4),
            "is_near_black_by_threshold": bool(is_black),
            "mean_absolute_difference_from_previous": round(diff_score, 4) if diff_score is not None else None,
            "possible_abrupt_change": bool(diff_score is not None and diff_score >= diff_threshold),
            "possible_duplicate_frame": bool(diff_score is not None and diff_score <= 0.25),
        })

        frame_index += 1
        if max_scan_frames is not None and frame_index >= max_scan_frames:
            break

    cap.release()

    # A contact sheet of sampled frames.
    contact_sheet = None
    if sampled_images:
        cols = 3
        cell_w, cell_h, label_h = 320, 200, 30
        rows = (len(sampled_images) + cols - 1) // cols
        contact_sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), "white")
        draw = ImageDraw.Draw(contact_sheet)
        for i, (idx, ts, im) in enumerate(sampled_images):
            x = (i % cols) * cell_w
            y = (i // cols) * (cell_h + label_h)
            im.thumbnail((cell_w - 12, cell_h - 12))
            contact_sheet.paste(im, (x + (cell_w - im.width) // 2, y + (cell_h - im.height) // 2))
            stamp = f"Frame {idx}" + (f" | {ts:.2f}s" if ts is not None else "")
            draw.text((x + 8, y + cell_h + 4), stamp, fill="black")

    changes = [r for r in frame_rows if r["possible_abrupt_change"]]
    near_black = [r for r in frame_rows if r["is_near_black_by_threshold"]]
    possible_duplicates = [r for r in frame_rows if r["possible_duplicate_frame"]]
    scanned_frames = len(frame_rows)

    metadata = {
        "file_name": Path(path).name,
        "file_size_bytes": os.path.getsize(path),
        "sha256": sha256_file(path),
        "container_or_codec": "Not determined by OpenCV; use ffprobe for deeper stream/container details.",
        "video_width": width,
        "video_height": height,
        "reported_fps": round(fps, 6) if fps else None,
        "reported_frame_count": frame_count_reported or None,
        "scanned_frame_count": scanned_frames,
        "estimated_duration_seconds": round(duration, 4) if duration is not None else None,
        "scan_was_limited": bool(max_scan_frames is not None and scanned_frames >= max_scan_frames),
    }
    summary = {
        "sample_interval_seconds": sample_every_seconds,
        "abrupt_change_threshold": diff_threshold,
        "near_black_mean_brightness_threshold": black_threshold,
        "scanned_frames": scanned_frames,
        "sampled_frames": sampled_count,
        "possible_abrupt_change_count": len(changes),
        "near_black_frame_count": len(near_black),
        "possible_duplicate_frame_count": len(possible_duplicates),
        "mean_frame_difference": round(float(np.mean(change_scores)), 4) if change_scores else None,
        "interpretation": (
            "These are heuristic indicators for human review, not proof of tampering. "
            "Fast motion, cuts, flashes, low-light footage, transcoding, and variable frame rate "
            "can produce similar patterns."
        ),
    }
    return metadata, summary, pd.DataFrame(frame_rows), sampled_images, contact_sheet


def build_html_report(report: dict) -> str:
    escaped = json.dumps(report, indent=2, ensure_ascii=False)
    safe = escaped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Video Forensics Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1000px;margin:32px auto;padding:0 20px;color:#202124}}
h1,h2{{color:#17365d}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:18px;border-radius:8px}}
.notice{{padding:12px;border-left:4px solid #d39e00;background:#fff8e1}}
</style></head><body>
<h1>Video Forensics Analyzer — Report</h1>
<p>Generated: {report["generated_at_utc"]}</p>
<div class="notice"><strong>Interpretation caution:</strong> Automated indicators are not proof that a video was manipulated. Validate findings with independent tools and context.</div>
<h2>Report data</h2><pre>{safe}</pre>
</body></html>"""


st.set_page_config(page_title="Video Forensics Analyzer", page_icon="🔎", layout="wide")
st.title("🔎 Video Forensics Analyzer")
st.caption("Local-first prototype for metadata inspection, frame anomaly triage, hashing, and evidence reporting.")
st.warning("Use this as an investigative aid, not an automatic authenticity verdict. Keep an untouched original and document every handling step.")

with st.sidebar:
    st.header("Analysis settings")
    sample_interval = st.slider("Sample frame every (seconds)", min_value=0.25, max_value=10.0, value=1.0, step=0.25)
    diff_threshold = st.slider("Abrupt-change threshold (mean pixel difference)", min_value=5.0, max_value=100.0, value=35.0, step=1.0)
    black_threshold = st.slider("Near-black brightness threshold (0–255)", min_value=0, max_value=40, value=8, step=1)
    max_frames = st.selectbox("Maximum frames to scan", options=[1000, 5000, 20000, 0], index=1,
                              format_func=lambda x: "All frames (may be slow)" if x == 0 else f"{x:,} frames")
    st.caption("Frame-difference thresholds are heuristic and depend on the footage.")

uploaded = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv", "m4v", "webm", "mpeg", "mpg"])
if uploaded:
    st.write(f"**File:** {uploaded.name} · **Size:** {uploaded.size:,} bytes")
    if st.button("Analyze video", type="primary"):
        suffix = Path(uploaded.name).suffix or ".video"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        try:
            with st.spinner("Scanning video frames..."):
                metadata, summary, frames_df, sampled_images, contact_sheet = analyze_video(
                    tmp_path, sample_interval, diff_threshold, black_threshold, max_frames
                )
            report = {
                "report_version": APP_VERSION,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "analysis_method": "OpenCV frame scan; SHA-256 of uploaded file bytes",
                "metadata": metadata,
                "summary": summary,
                "frame_findings": frames_df.to_dict(orient="records"),
            }
            st.session_state["vf_report"] = report
            st.session_state["vf_frames"] = frames_df
            st.session_state["vf_samples"] = sampled_images
            st.session_state["vf_contact"] = contact_sheet
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

if "vf_report" in st.session_state:
    report = st.session_state["vf_report"]
    metadata = report["metadata"]
    summary = report["summary"]
    st.subheader("Analysis summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames scanned", f'{summary["scanned_frames"]:,}')
    c2.metric("Abrupt-change flags", f'{summary["possible_abrupt_change_count"]:,}')
    c3.metric("Near-black frames", f'{summary["near_black_frame_count"]:,}')
    c4.metric("Duplicate-like frames", f'{summary["possible_duplicate_frame_count"]:,}')

    st.subheader("File metadata and integrity")
    st.json(metadata)
    st.info(summary["interpretation"])

    st.subheader("Frame findings")
    frames_df = st.session_state["vf_frames"]
    flagged = frames_df[
        frames_df["possible_abrupt_change"] |
        frames_df["is_near_black_by_threshold"] |
        frames_df["possible_duplicate_frame"]
    ]
    st.dataframe(flagged, use_container_width=True, hide_index=True)
    st.download_button("Download all frame findings (CSV)", frames_df.to_csv(index=False).encode("utf-8"),
                       file_name="frame_findings.csv", mime="text/csv")

    st.subheader("Sampled frames")
    samples = st.session_state["vf_samples"]
    if samples:
        cols = st.columns(3)
        for i, (idx, ts, im) in enumerate(samples[:60]):
            with cols[i % 3]:
                st.image(im, caption=f"Frame {idx}" + (f" · {ts:.2f}s" if ts is not None else ""), use_container_width=True)
        if len(samples) > 60:
            st.caption("Only the first 60 sampled frames are displayed; the report contains frame-level findings for all scanned frames.")

    contact = st.session_state.get("vf_contact")
    if contact:
        st.subheader("Contact sheet")
        st.image(contact, caption="Sampled frames overview", use_container_width=True)

    st.subheader("Export evidence report")
    json_bytes = json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")
    html_bytes = build_html_report(report).encode("utf-8")
    st.download_button("Download JSON report", json_bytes, file_name="video_forensics_report.json", mime="application/json")
    st.download_button("Download HTML report", html_bytes, file_name="video_forensics_report.html", mime="text/html")

st.divider()
st.caption("Privacy note: the uploaded file is processed by this Streamlit session and temporary local storage. For sensitive evidence, run locally on an isolated machine and follow your organization's evidence-handling policy.")
