#!/usr/bin/env python3
"""Independent MediaPipe face-landmark validation for AcousticPose proxies.

This is intentionally a small, CPU-safe validation pass. It samples clips from
the existing indexed video datasets, estimates simple face-landmark motion
proxies, resizes them to 160 frames, and compares them against the cached
optical-flow proxy targets used by AcousticPose.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy import stats
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import acousticpose_local as ap


TARGET_DIMS = ["head_yaw", "head_pitch", "head_roll", "torso_lean", "motion_energy"]
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def cache_key(dataset: str, clip_id: str) -> str:
    return hashlib.md5(f"{dataset}_{clip_id}".encode()).hexdigest()


def resize_seq(arr: np.ndarray, target_len: int = 160) -> np.ndarray:
    if len(arr) == 0:
        return np.zeros((target_len, 5), np.float32)
    arr = np.asarray(arr, np.float32)
    if len(arr) == target_len:
        return arr
    old = np.linspace(0, 1, len(arr))
    new = np.linspace(0, 1, target_len)
    cols = [np.interp(new, old, arr[:, i]) for i in range(arr.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    s = float(np.nanstd(x))
    if s < 1e-8:
        return np.zeros_like(x)
    return (x - float(np.nanmean(x))) / s


def corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = zscore(a)
    b = zscore(b)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0, 0.0
    pearson = float(np.corrcoef(a, b)[0, 1])
    spearman = float(stats.spearmanr(a, b).correlation)
    if not np.isfinite(pearson):
        pearson = 0.0
    if not np.isfinite(spearman):
        spearman = 0.0
    return pearson, spearman


def ensure_model(out_dir: Path) -> Path:
    model_path = out_dir / "face_landmarker.task"
    if not model_path.exists():
        print("download", MODEL_URL, flush=True)
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def mediapipe_proxy(video_path: Path, model_path: Path, target_len: int = 160, max_frames: int = 220) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // max_frames) if total else 1
    rows = []
    seen = 0
    misses = 0
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step:
                idx += 1
                continue
            idx += 1
            seen += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(1000 * idx / max(cap.get(cv2.CAP_PROP_FPS) or 25, 1))
            result = landmarker.detect_for_video(image, timestamp_ms)
            if not result.face_landmarks:
                rows.append([0, 0, 0, 0, 0])
                misses += 1
                continue
            lm = result.face_landmarks[0]
            # Stable approximate facial landmarks in normalized image coordinates.
            nose = lm[1]
            left_eye = lm[33]
            right_eye = lm[263]
            chin = lm[152]
            forehead = lm[10]
            mouth_l = lm[61]
            mouth_r = lm[291]
            eye_cx = (left_eye.x + right_eye.x) / 2
            eye_cy = (left_eye.y + right_eye.y) / 2
            face_w = max(abs(right_eye.x - left_eye.x), 1e-4)
            face_h = max(abs(chin.y - forehead.y), 1e-4)
            yaw = (nose.x - eye_cx) / face_w
            pitch = (nose.y - eye_cy) / face_h
            roll = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
            lean = ((mouth_l.x + mouth_r.x) / 2 - eye_cx) / face_w
            rows.append([yaw, pitch, roll, lean, 0.0])
    cap.release()
    arr = np.asarray(rows, np.float32)
    if len(arr) > 1:
        motion = np.linalg.norm(np.diff(arr[:, :4], axis=0), axis=1)
        arr[1:, 4] = motion
        arr[0, 4] = motion[0] if len(motion) else 0.0
    miss_rate = misses / max(seen, 1)
    return resize_seq(arr, target_len), miss_rate


def build_index(project: Path) -> pd.DataFrame:
    frames = []
    data = project / "data"
    for name, fn in [
        ("RAVDESS", lambda: ap.index_ravdess(data / "RAVDESS")),
        ("CREMA-D", lambda: ap.index_cremad(data / "CREMA-D")),
        ("MELD", lambda: ap.index_meld(data / "MELD")),
    ]:
        try:
            df = fn()
            if len(df):
                frames.append(df)
        except Exception as exc:
            print("index skip", name, repr(exc), flush=True)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("work/full_public_stage1"))
    parser.add_argument("--per-dataset", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/full_public_stage1_results"))
    args = parser.parse_args()

    project = args.project.resolve()
    cache_y = project / "cache" / "Y"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = ensure_model(args.out_dir)
    index = build_index(project)
    rows = []
    rng = np.random.default_rng(args.seed)
    for dataset, group in index.groupby("dataset"):
        candidates = []
        for _, row in group.iterrows():
            y_path = cache_y / f"{cache_key(row.dataset, row.clip_id)}.npy"
            video_path = Path(str(row.video_path))
            if y_path.exists() and video_path.exists():
                candidates.append((row, y_path, video_path))
        if not candidates:
            continue
        chosen = rng.choice(len(candidates), size=min(args.per_dataset, len(candidates)), replace=False)
        for i in chosen:
            item, y_path, video_path = candidates[int(i)]
            print("validate", dataset, item.clip_id, flush=True)
            mp_proxy, miss = mediapipe_proxy(video_path, model_path)
            target = np.load(y_path).astype(np.float32)
            target = resize_seq(target)
            metrics = {
                "dataset": dataset,
                "clip_id": item.clip_id,
                "video_path": str(video_path),
                "miss_rate": miss,
            }
            for j, name in enumerate(TARGET_DIMS):
                p, s = corr(mp_proxy[:, j], target[:, j])
                metrics[f"{name}_pearson"] = p
                metrics[f"{name}_spearman"] = s
            p, s = corr(mp_proxy[:, 4], target[:, 4])
            metrics["motion_energy_pearson"] = p
            metrics["motion_energy_spearman"] = s
            rows.append(metrics)
            pd.DataFrame(rows).to_csv(args.out_dir / "mediapipe_headpose_proxy_validation.csv", index=False)

    out = pd.DataFrame(rows)
    if len(out):
        summary = out.groupby("dataset").agg(
            clips=("clip_id", "count"),
            miss_rate=("miss_rate", "mean"),
            energy_spearman=("motion_energy_spearman", "median"),
            yaw_spearman=("head_yaw_spearman", "median"),
            pitch_spearman=("head_pitch_spearman", "median"),
            roll_spearman=("head_roll_spearman", "median"),
            lean_spearman=("torso_lean_spearman", "median"),
        ).reset_index()
        summary.to_csv(args.out_dir / "mediapipe_headpose_proxy_validation_summary.csv", index=False)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
