#!/usr/bin/env python3
"""Train and visualize an audio-to-video-landmark motion proof head.

This experiment is deliberately scoped to video-reference facial landmark
motion. MediaPipe landmarks provide the reference trajectory; the regressor
predicts landmark *displacements* from audio features only. For visualization,
the clip's neutral face shape is reused only as a display scaffold, while all
temporal displacement comes from audio.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import acousticpose_local as ap
import mediapipe_headpose_validation as mpv


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "work/full_public_stage1"
RESULTS = ROOT / "outputs/full_public_stage1_results"
OUT = RESULTS / "landmark_reconstruction"
LANDMARK_CACHE = OUT / "landmark_cache"
SITE = ROOT / "outputs/website"
SITE_DEMOS = SITE / "assets/demos"
SITE_FIGS = SITE / "assets/figures"
OVERLEAF_FIGS = ROOT / "outputs/overleaf/figures"

FPS = 25
TARGET_LEN = 160
MAX_EXTRACT_FRAMES = 120
TRAIN_LIMIT = 240
TEST_LIMIT = 80
DEMO_LIMIT = 5

# Dense privacy-preserving MediaPipe subset: enough points to read expression and
# head motion, but still rendered as a generic white landmark avatar.
LANDMARK_GROUPS = {
    "face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "left_brow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_brow": [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
    "left_eye": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    "right_eye": [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249],
    "nose": [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327],
    "outer_lip": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78],
    "inner_lip": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    "mid_face": [205, 50, 101, 36, 206, 207, 187, 123, 425, 280, 330, 266, 426, 427, 411, 352],
}
POINT_IDS = list(dict.fromkeys([idx for group in LANDMARK_GROUPS.values() for idx in group]))
POINT_NAMES = [f"mp_{idx}" for idx in POINT_IDS]
POINT_INDEX = {idx: i for i, idx in enumerate(POINT_IDS)}
LANDMARK_CACHE_VERSION = "dense_v2"


def point_indices(ids: list[int]) -> np.ndarray:
    return np.asarray([POINT_INDEX[i] for i in ids if i in POINT_INDEX], dtype=int)


LOWER_FACE_INDICES = point_indices([152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324])
MOUTH_INDICES = point_indices(LANDMARK_GROUPS["outer_lip"] + LANDMARK_GROUPS["inner_lip"])
BROW_INDICES = point_indices(LANDMARK_GROUPS["left_brow"] + LANDMARK_GROUPS["right_brow"])
EYE_INDICES = point_indices(LANDMARK_GROUPS["left_eye"] + LANDMARK_GROUPS["right_eye"])
NOSE_INDICES = point_indices(LANDMARK_GROUPS["nose"])
FACE_OVAL_INDICES = point_indices(LANDMARK_GROUPS["face_oval"])


def build_connections() -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    ring_groups = ["face_oval", "left_eye", "right_eye", "outer_lip", "inner_lip"]
    chain_groups = ["left_brow", "right_brow", "nose"]
    for name in ring_groups:
        ids = LANDMARK_GROUPS[name]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            if a in POINT_INDEX and b in POINT_INDEX:
                pairs.append((POINT_INDEX[a], POINT_INDEX[b]))
    for name in chain_groups:
        ids = LANDMARK_GROUPS[name]
        for a, b in zip(ids[:-1], ids[1:]):
            if a in POINT_INDEX and b in POINT_INDEX:
                pairs.append((POINT_INDEX[a], POINT_INDEX[b]))
    cross = [(10, 168), (168, 6), (6, 1), (1, 13), (13, 14), (14, 17), (61, 291), (33, 133), (263, 362), (234, 454), (152, 17), (205, 425), (50, 280), (101, 330)]
    for a, b in cross:
        if a in POINT_INDEX and b in POINT_INDEX:
            pairs.append((POINT_INDEX[a], POINT_INDEX[b]))
    return list(dict.fromkeys(pairs))


CONNECTIONS = build_connections()
REF_COLOR = (80, 210, 80)      # OpenCV BGR: green video-reference landmarks.
PRED_COLOR = (50, 140, 245)    # OpenCV BGR: orange audio-only reconstruction.


def configure() -> None:
    ap.CFG = ap.Config(
        project_root=PROJECT,
        data_root=PROJECT / "data",
        cache_root=PROJECT / "cache",
        output_root=PROJECT / "outputs",
        sota_root=PROJECT / "sota_outputs",
        epochs=3,
        patience=1,
        batch_size=32,
        device="cpu",
        target_backend="optical_flow",
        hidden_size=32,
    )
    ap.seed_everything(42)
    torch.set_num_threads(2)
    for path in [OUT, LANDMARK_CACHE, SITE_DEMOS, SITE_FIGS, OVERLEAF_FIGS]:
        path.mkdir(parents=True, exist_ok=True)


def cache_key(dataset: str, clip_id: str) -> str:
    return hashlib.md5(f"{LANDMARK_CACHE_VERSION}_{dataset}_{clip_id}".encode()).hexdigest()


def resize_seq(arr: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    arr = np.asarray(arr, np.float32)
    if len(arr) == target_len:
        return arr
    if len(arr) == 0:
        return np.zeros((target_len,) + arr.shape[1:], np.float32)
    old = np.linspace(0, 1, len(arr))
    new = np.linspace(0, 1, target_len)
    flat = arr.reshape(len(arr), -1)
    out = np.stack([np.interp(new, old, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return out.reshape((target_len,) + arr.shape[1:]).astype(np.float32)


def interp_missing(arr: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(arr, np.float32)
    missing = ~np.isfinite(arr).all(axis=(1, 2))
    miss_rate = float(missing.mean()) if len(missing) else 1.0
    flat = arr.reshape(len(arr), -1)
    x = np.arange(len(flat))
    for j in range(flat.shape[1]):
        col = flat[:, j]
        good = np.isfinite(col)
        if good.sum() == 0:
            col[:] = 0
        elif good.sum() == 1:
            col[:] = col[good][0]
        else:
            col[~good] = np.interp(x[~good], x[good], col[good])
        flat[:, j] = col
    return flat.reshape(arr.shape).astype(np.float32), miss_rate


def canonicalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    left = points[:, POINT_IDS.index(33), :2]
    right = points[:, POINT_IDS.index(263), :2]
    eye_center = (left + right) * 0.5
    scale = np.linalg.norm(right - left, axis=1)
    scale = np.where(scale < 1e-4, np.nanmedian(scale[scale > 1e-4]) if np.any(scale > 1e-4) else 0.12, scale)
    xy = (points[:, :, :2] - eye_center[:, None, :]) / scale[:, None, None]
    z = points[:, :, 2:3] / scale[:, None, None]
    canonical = np.concatenate([xy, z], axis=2).astype(np.float32)
    neutral = np.median(canonical, axis=0).astype(np.float32)
    return canonical, neutral, float(np.nanmedian(scale))


def extract_landmarks(video_path: Path, dataset: str, clip_id: str, model_path: Path) -> dict:
    out_path = LANDMARK_CACHE / f"{cache_key(dataset, clip_id)}.npz"
    if out_path.exists():
        data = np.load(out_path)
        return {
            "canonical": data["canonical"],
            "neutral": data["neutral"],
            "miss_rate": float(data["miss_rate"]),
            "raw": data["raw"],
        }
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // MAX_EXTRACT_FRAMES) if total else 1
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    rows: list[np.ndarray] = []
    misses = 0
    seen = 0
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
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
            timestamp_ms = int(1000 * idx / max(fps, 1))
            idx += 1
            seen += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, timestamp_ms)
            if not result.face_landmarks:
                rows.append(np.full((len(POINT_IDS), 3), np.nan, np.float32))
                misses += 1
                continue
            lm = result.face_landmarks[0]
            pts = np.asarray([[lm[i].x, lm[i].y, lm[i].z] for i in POINT_IDS], np.float32)
            rows.append(pts)
    cap.release()
    if not rows:
        raw = np.full((TARGET_LEN, len(POINT_IDS), 3), np.nan, np.float32)
    else:
        raw = resize_seq(np.asarray(rows, np.float32), TARGET_LEN)
    raw, miss_rate_interp = interp_missing(raw)
    miss_rate = max(misses / max(seen, 1), miss_rate_interp)
    canonical, neutral, _ = canonicalize(raw)
    np.savez_compressed(out_path, raw=raw, canonical=canonical, neutral=neutral, miss_rate=np.asarray(miss_rate, np.float32))
    return {"canonical": canonical, "neutral": neutral, "miss_rate": miss_rate, "raw": raw}


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_index = pd.read_csv(PROJECT / "outputs/tables/real_feature_index.csv")
    train_df, val_df, test_df = ap.split_by_speaker(feature_index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def candidate_rows(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = pd.read_csv(RESULTS / "mediapipe_headpose_large_validation.csv")
    validation = validation[(validation.detector_success_rate >= 0.95) & (validation.miss_rate <= 0.05)]
    validation = validation.sort_values(["spearman_motion_energy", "detector_success_rate"], ascending=[False, False])
    feature_index = pd.read_csv(PROJECT / "outputs/tables/real_feature_index.csv")
    merged = validation[["dataset", "clip_id", "spearman_motion_energy", "detector_success_rate"]].merge(
        feature_index,
        on=["dataset", "clip_id"],
        how="inner",
    )
    merged = merged[merged.video_path.map(lambda p: Path(str(p)).exists())].reset_index(drop=True)
    speakers = np.asarray(sorted(merged.speaker_id.astype(str).unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(speakers)
    split = max(1, int(len(speakers) * 0.72))
    train_speakers = set(speakers[:split])
    train = merged[merged.speaker_id.astype(str).isin(train_speakers)].copy()
    test = merged[~merged.speaker_id.astype(str).isin(train_speakers)].copy()
    # Keep datasets balanced enough for a quick CPU proof pass.
    train = (
        train.sort_values(["dataset", "spearman_motion_energy"], ascending=[True, False])
        .groupby("dataset", group_keys=False)
        .head(max(20, TRAIN_LIMIT // 3))
        .head(TRAIN_LIMIT)
        .reset_index(drop=True)
    )
    test = (
        test.sort_values(["dataset", "spearman_motion_energy"], ascending=[True, False])
        .groupby("dataset", group_keys=False)
        .head(max(12, TEST_LIMIT // 3))
        .head(TEST_LIMIT)
        .reset_index(drop=True)
    )
    return train, test


def build_arrays(rows: pd.DataFrame, model_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    xs, ys, neutrals, meta = [], [], [], []
    for _, row in rows.iterrows():
        item = extract_landmarks(Path(row.video_path), str(row.dataset), str(row.clip_id), model_path)
        if item["miss_rate"] > 0.08:
            continue
        x = np.load(row.feature_path).astype(np.float32)
        x = resize_seq(x, TARGET_LEN)
        canonical = item["canonical"].astype(np.float32)
        neutral = item["neutral"].astype(np.float32)
        motion = canonical - neutral[None, :, :]
        xs.append(x)
        ys.append(motion.reshape(TARGET_LEN, -1))
        neutrals.append(neutral.reshape(-1))
        meta.append(
            {
                "dataset": str(row.dataset),
                "clip_id": str(row.clip_id),
                "speaker_id": str(row.speaker_id),
                "video_path": str(row.video_path),
                "audio_path": str(row.audio_path),
                "miss_rate": float(item["miss_rate"]),
            }
        )
        print("landmarks", row.dataset, row.clip_id, "miss", f"{item['miss_rate']:.3f}", flush=True)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(neutrals, np.float32), meta


def mouth_open(arr: np.ndarray) -> np.ndarray:
    pts = arr.reshape(arr.shape[0], len(POINT_IDS), 3)
    upper = pts[:, POINT_IDS.index(13), :2]
    lower = pts[:, POINT_IDS.index(14), :2]
    return np.linalg.norm(upper - lower, axis=1)


def point_xy(pts: np.ndarray, landmark_id: int) -> np.ndarray:
    return pts[:, POINT_INDEX[landmark_id], :2]


def lip_spread(arr: np.ndarray) -> np.ndarray:
    pts = arr.reshape(arr.shape[0], len(POINT_IDS), 3)
    return np.linalg.norm(point_xy(pts, 61) - point_xy(pts, 291), axis=1)


def jaw_drop(arr: np.ndarray) -> np.ndarray:
    pts = arr.reshape(arr.shape[0], len(POINT_IDS), 3)
    return np.linalg.norm(point_xy(pts, 14) - point_xy(pts, 152), axis=1)


def brow_lift(arr: np.ndarray) -> np.ndarray:
    pts = arr.reshape(arr.shape[0], len(POINT_IDS), 3)
    left = point_xy(pts, 159)[:, 1] - point_xy(pts, 105)[:, 1]
    right = point_xy(pts, 386)[:, 1] - point_xy(pts, 334)[:, 1]
    return (left + right) * 0.5


def eye_open(arr: np.ndarray) -> np.ndarray:
    pts = arr.reshape(arr.shape[0], len(POINT_IDS), 3)
    left = np.linalg.norm(point_xy(pts, 159) - point_xy(pts, 145), axis=1)
    right = np.linalg.norm(point_xy(pts, 386) - point_xy(pts, 374), axis=1)
    return (left + right) * 0.5


def motion_energy(arr: np.ndarray) -> np.ndarray:
    pts = arr.reshape(arr.shape[0], len(POINT_IDS), 3)
    diff = np.diff(pts[:, :, :2], axis=0, prepend=pts[:1, :, :2])
    return np.linalg.norm(diff, axis=2).mean(axis=1)


def case_aspects(y_motion: np.ndarray, neutral: np.ndarray) -> dict[str, np.ndarray]:
    geom = y_motion + neutral[None, :]
    return {
        "mouth aperture": mouth_open(geom),
        "lip spread": lip_spread(geom),
        "jaw drop": jaw_drop(geom),
        "brow lift": brow_lift(geom),
        "eye openness": eye_open(geom),
        "motion energy": motion_energy(y_motion),
    }


def draw_trace_grid(canvas: np.ndarray, true_aspects: dict[str, np.ndarray], pred_aspects: dict[str, np.ndarray], labels: list[str]) -> None:
    canvas[:] = (255, 255, 255)
    h, w = canvas.shape[:2]
    cols = 3
    rows = int(math.ceil(len(labels) / cols))
    cell_w = w // cols
    cell_h = h // rows
    for idx, label in enumerate(labels):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_w + 20
        y0 = r * cell_h + 28
        x1 = (c + 1) * cell_w - 24
        y1 = (r + 1) * cell_h - 18
        cv2.putText(canvas, label, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (50, 56, 62), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (232, 235, 230), 1, cv2.LINE_AA)
        for vals, color in [(true_aspects[label], REF_COLOR), (pred_aspects[label], PRED_COLOR)]:
            vals = np.asarray(vals, np.float32)
            vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            pts_line = []
            for j, v in enumerate(vals):
                x = x0 + 8 + int(j / max(1, len(vals) - 1) * (x1 - x0 - 16))
                y = y1 - 8 - int(v * (y1 - y0 - 18))
                pts_line.append((x, y))
            for a, b in zip(pts_line[:-1], pts_line[1:]):
                cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-7 or np.std(b) < 1e-7:
        return 0.0
    val = pearsonr(a, b).statistic
    return float(0.0 if not np.isfinite(val) else val)


def event_f1(a: np.ndarray, b: np.ndarray) -> float:
    def peaks(v: np.ndarray) -> set[int]:
        prom = float(np.std(v) * 0.35) if np.std(v) > 1e-8 else 0.01
        ids, _ = find_peaks(v, distance=8, prominence=prom)
        return set(map(int, ids))

    pa, pb = peaks(a), peaks(b)
    if not pa and not pb:
        return 1.0
    if not pa or not pb:
        return 0.0
    matched = 0
    used: set[int] = set()
    for p in pa:
        close = [q for q in pb if abs(q - p) <= 5 and q not in used]
        if close:
            q = min(close, key=lambda x: abs(x - p))
            used.add(q)
            matched += 1
    precision = matched / max(len(pb), 1)
    recall = matched / max(len(pa), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-8))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, neutral: np.ndarray, meta: list[dict]) -> pd.DataFrame:
    rows = []
    for i in range(len(y_true)):
        true = y_true[i]
        pred = y_pred[i]
        zero = np.zeros_like(true)
        pts_true = true.reshape(TARGET_LEN, len(POINT_IDS), 3)
        pts_pred = pred.reshape(TARGET_LEN, len(POINT_IDS), 3)
        nme = np.linalg.norm(pts_true[:, :, :2] - pts_pred[:, :, :2], axis=2).mean(axis=1)
        zero_nme = np.linalg.norm(pts_true[:, :, :2] - zero.reshape(TARGET_LEN, len(POINT_IDS), 3)[:, :, :2], axis=2).mean(axis=1)
        lower_nme = np.linalg.norm(pts_true[:, LOWER_FACE_INDICES, :2] - pts_pred[:, LOWER_FACE_INDICES, :2], axis=2).mean(axis=1)
        lower_zero_nme = np.linalg.norm(pts_true[:, LOWER_FACE_INDICES, :2], axis=2).mean(axis=1)
        mouth_nme = np.linalg.norm(pts_true[:, MOUTH_INDICES, :2] - pts_pred[:, MOUTH_INDICES, :2], axis=2).mean(axis=1)
        mouth_zero_nme = np.linalg.norm(pts_true[:, MOUTH_INDICES, :2], axis=2).mean(axis=1)
        brow_nme = np.linalg.norm(pts_true[:, BROW_INDICES, :2] - pts_pred[:, BROW_INDICES, :2], axis=2).mean(axis=1)
        eye_nme = np.linalg.norm(pts_true[:, EYE_INDICES, :2] - pts_pred[:, EYE_INDICES, :2], axis=2).mean(axis=1)
        true_mouth = mouth_open(true + neutral[i][None, :])
        pred_mouth = mouth_open(pred + neutral[i][None, :])
        true_spread = lip_spread(true + neutral[i][None, :])
        pred_spread = lip_spread(pred + neutral[i][None, :])
        true_jaw = jaw_drop(true + neutral[i][None, :])
        pred_jaw = jaw_drop(pred + neutral[i][None, :])
        true_brow = brow_lift(true + neutral[i][None, :])
        pred_brow = brow_lift(pred + neutral[i][None, :])
        true_eye = eye_open(true + neutral[i][None, :])
        pred_eye = eye_open(pred + neutral[i][None, :])
        true_energy = motion_energy(true)
        pred_energy = motion_energy(pred)
        aspect_corrs = [
            safe_corr(true_mouth, pred_mouth),
            safe_corr(true_spread, pred_spread),
            safe_corr(true_jaw, pred_jaw),
            safe_corr(true_brow, pred_brow),
            safe_corr(true_eye, pred_eye),
            safe_corr(true_energy, pred_energy),
        ]
        rows.append(
            {
                **meta[i],
                "landmark_motion_nme": float(np.mean(nme)),
                "static_neutral_nme": float(np.mean(zero_nme)),
                "relative_nme_improvement": float((np.mean(zero_nme) - np.mean(nme)) / max(np.mean(zero_nme), 1e-8)),
                "lower_face_nme": float(np.mean(lower_nme)),
                "lower_face_static_nme": float(np.mean(lower_zero_nme)),
                "lower_face_relative_improvement": float((np.mean(lower_zero_nme) - np.mean(lower_nme)) / max(np.mean(lower_zero_nme), 1e-8)),
                "mouth_region_nme": float(np.mean(mouth_nme)),
                "mouth_region_static_nme": float(np.mean(mouth_zero_nme)),
                "mouth_region_relative_improvement": float((np.mean(mouth_zero_nme) - np.mean(mouth_nme)) / max(np.mean(mouth_zero_nme), 1e-8)),
                "brow_region_nme": float(np.mean(brow_nme)),
                "eye_region_nme": float(np.mean(eye_nme)),
                "lower_face_pck_10pct": float(np.mean(lower_nme < 0.10)),
                "mouth_region_pck_10pct": float(np.mean(mouth_nme < 0.10)),
                "brow_region_pck_10pct": float(np.mean(brow_nme < 0.10)),
                "eye_region_pck_10pct": float(np.mean(eye_nme < 0.10)),
                "pck_10pct": float(np.mean(nme < 0.10)),
                "pck_15pct": float(np.mean(nme < 0.15)),
                "mouth_open_corr": safe_corr(true_mouth, pred_mouth),
                "lip_spread_corr": safe_corr(true_spread, pred_spread),
                "jaw_drop_corr": safe_corr(true_jaw, pred_jaw),
                "brow_lift_corr": safe_corr(true_brow, pred_brow),
                "eye_open_corr": safe_corr(true_eye, pred_eye),
                "landmark_energy_corr": safe_corr(true_energy, pred_energy),
                "multi_aspect_corr": float(np.mean([max(v, 0.0) for v in aspect_corrs])),
                "landmark_event_f1": event_f1(true_energy, pred_energy),
                "articulatory_score": float(
                    0.22 * np.mean(mouth_nme < 0.10)
                    + 0.16 * np.mean(lower_nme < 0.10)
                    + 0.16 * np.mean(brow_nme < 0.10)
                    + 0.16 * np.mean(eye_nme < 0.10)
                    + 0.20 * np.mean([max(v, 0.0) for v in aspect_corrs])
                    + 0.10 * event_f1(true_energy, pred_energy)
                ),
            }
        )
    return pd.DataFrame(rows)


def draw_landmarks(frame: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], alpha: float = 1.0) -> None:
    overlay = frame.copy()
    h, w = frame.shape[:2]
    xy = pts[:, :2].copy()
    draw_pts = np.column_stack([xy[:, 0] * w, xy[:, 1] * h]).astype(int)
    dense = len(POINT_IDS) > 60
    bg_thick = 2 if dense else 4
    fg_thick = 1 if dense else 2
    outer_radius = 2 if dense else 5
    inner_radius = 1 if dense else 3
    for a, b in CONNECTIONS:
        cv2.line(overlay, tuple(draw_pts[a]), tuple(draw_pts[b]), (16, 18, 22), bg_thick, cv2.LINE_AA)
        cv2.line(overlay, tuple(draw_pts[a]), tuple(draw_pts[b]), color, fg_thick, cv2.LINE_AA)
    for p in draw_pts:
        cv2.circle(overlay, tuple(p), outer_radius, (16, 18, 22), -1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(p), inner_radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def project_3d_points(pts: np.ndarray, size: int, yaw_deg: float = -22.0, pitch_deg: float = 8.0, scale: float = 135.0) -> np.ndarray:
    """Project canonical xyz face points to a compact orthographic 3D view."""
    pts = pts.astype(np.float32).copy()
    pts -= np.median(pts, axis=0, keepdims=True)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    ry = np.asarray(
        [
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ],
        np.float32,
    )
    rx = np.asarray(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ],
        np.float32,
    )
    rot = pts @ ry.T @ rx.T
    # MediaPipe image coordinates already use positive-y downward. Keeping that
    # sign here makes the generic 3D landmark avatar read upright.
    xy = np.column_stack([rot[:, 0], rot[:, 1]])
    xy = xy * scale + np.asarray([size * 0.5, size * 0.48], np.float32)
    depth = rot[:, 2]
    return np.column_stack([xy, depth])


def draw_3d_reconstruction(canvas: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], title: str, subtitle: str = "") -> None:
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (222, 226, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (32, 38, 42), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (75, 84, 90), 1, cv2.LINE_AA)
    view = np.full((h - 90, w - 28, 3), (241, 242, 236), np.uint8)
    proj = project_3d_points(pts, min(view.shape[:2]), yaw_deg=-18, pitch_deg=6, scale=min(view.shape[:2]) * 0.31)
    xy = proj[:, :2].astype(int)
    z = proj[:, 2]
    order = np.argsort(z)
    center = (view.shape[1] // 2, int(view.shape[0] * 0.48))
    head_axes = (int(view.shape[1] * 0.27), int(view.shape[0] * 0.38))
    cv2.ellipse(view, center, head_axes, -8, 0, 360, (251, 251, 248), -1, cv2.LINE_AA)
    cv2.ellipse(view, center, head_axes, -8, 0, 360, (205, 210, 205), 2, cv2.LINE_AA)
    cv2.ellipse(view, (center[0], center[1] + int(head_axes[1] * 0.12)), (int(head_axes[0] * 0.82), int(head_axes[1] * 0.52)), -8, 0, 360, (232, 235, 231), 1, cv2.LINE_AA)
    cv2.line(view, (center[0], center[1] - head_axes[1]), (center[0], center[1] + head_axes[1]), (228, 231, 226), 1, cv2.LINE_AA)
    for a, b in CONNECTIONS:
        shade = int(38 + 50 * np.clip((z[a] + z[b]) * 0.5 + 0.5, 0, 1))
        cv2.line(view, tuple(xy[a]), tuple(xy[b]), (shade, shade, shade), 3, cv2.LINE_AA)
        cv2.line(view, tuple(xy[a]), tuple(xy[b]), color, 1, cv2.LINE_AA)
    for idx in order:
        radius = int(2 + 2 * np.clip(z[idx] + 0.5, 0, 1))
        cv2.circle(view, tuple(xy[idx]), radius + 1, (26, 30, 32), -1, cv2.LINE_AA)
        cv2.circle(view, tuple(xy[idx]), radius, color, -1, cv2.LINE_AA)
        if radius >= 3:
            cv2.circle(view, tuple(xy[idx] - np.asarray([1, 1])), 1, (245, 248, 246), -1, cv2.LINE_AA)
    cv2.ellipse(view, (view.shape[1] // 2, int(view.shape[0] * 0.83)), (130, 24), 0, 0, 360, (218, 220, 214), 2, cv2.LINE_AA)
    canvas[78 : 78 + view.shape[0], 14 : 14 + view.shape[1]] = view


def draw_3d_video_aligned(canvas: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], title: str, subtitle: str = "") -> None:
    """Draw a generic 3D landmark avatar in the same x/y coordinate frame as video.

    Unlike the canonical renderer above, this preserves the raw MediaPipe image
    geometry, so the reference panel visually agrees with the video overlay.
    Depth is used for point ordering and size, while identity texture is omitted.
    """
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (222, 226, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (32, 38, 42), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (75, 84, 90), 1, cv2.LINE_AA)

    view = np.full((h - 90, w - 28, 3), (241, 242, 236), np.uint8)
    pts = np.asarray(pts, np.float32)
    xy_raw = pts[:, :2].copy()
    valid = np.isfinite(xy_raw).all(axis=1)
    if valid.sum() < 3:
        canvas[78 : 78 + view.shape[0], 14 : 14 + view.shape[1]] = view
        return

    anchor_idx = FACE_OVAL_INDICES if len(FACE_OVAL_INDICES) else np.arange(len(POINT_IDS))
    anchor = xy_raw[anchor_idx]
    anchor = anchor[np.isfinite(anchor).all(axis=1)]
    if len(anchor) < 3:
        anchor = xy_raw[valid]
    lo = np.percentile(anchor, 2, axis=0)
    hi = np.percentile(anchor, 98, axis=0)
    center_raw = (lo + hi) * 0.5
    span = np.maximum(hi - lo, 1e-4)
    scale = 0.76 * min(view.shape[1] / span[0], view.shape[0] / span[1])
    xy = (xy_raw - center_raw[None, :]) * scale + np.asarray([view.shape[1] * 0.5, view.shape[0] * 0.50], np.float32)
    draw_pts = xy.astype(int)

    z = pts[:, 2].copy()
    finite_z = z[np.isfinite(z)]
    if len(finite_z):
        z = (z - np.nanmedian(finite_z)) / (np.nanpercentile(finite_z, 90) - np.nanpercentile(finite_z, 10) + 1e-6)
    else:
        z = np.zeros(len(pts), np.float32)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    oval_pts = draw_pts[FACE_OVAL_INDICES] if len(FACE_OVAL_INDICES) else draw_pts[valid]
    oval_pts = oval_pts[np.isfinite(oval_pts).all(axis=1)]
    if len(oval_pts) >= 3:
        x, y, ww, hh = cv2.boundingRect(oval_pts.astype(np.int32))
        center = (x + ww // 2, y + hh // 2)
        axes = (max(24, int(ww * 0.56)), max(32, int(hh * 0.57)))
        angle = 0.0
        if len(oval_pts) >= 5:
            (_, _), (fit_w, fit_h), fit_angle = cv2.fitEllipse(oval_pts.astype(np.float32))
            # OpenCV swaps axes depending on fit orientation; this keeps the
            # generic head scaffold aligned with the observed in-plane face roll.
            angle = float(fit_angle - 90.0 if fit_h >= fit_w else fit_angle)
    else:
        center = (view.shape[1] // 2, view.shape[0] // 2)
        axes = (int(view.shape[1] * 0.28), int(view.shape[0] * 0.38))
        angle = 0.0
    cv2.ellipse(view, center, axes, angle, 0, 360, (252, 252, 249), -1, cv2.LINE_AA)
    cv2.ellipse(view, center, axes, angle, 0, 360, (204, 210, 205), 2, cv2.LINE_AA)
    cv2.ellipse(view, (center[0], center[1] + int(axes[1] * 0.10)), (int(axes[0] * 0.84), int(axes[1] * 0.52)), angle, 0, 360, (232, 235, 231), 1, cv2.LINE_AA)
    if POINT_INDEX.get(10) is not None and POINT_INDEX.get(152) is not None:
        cv2.line(view, tuple(draw_pts[POINT_INDEX[10]]), tuple(draw_pts[POINT_INDEX[152]]), (228, 231, 226), 1, cv2.LINE_AA)

    for a, b in CONNECTIONS:
        if not (valid[a] and valid[b]):
            continue
        shade = int(40 + 45 * np.clip(0.5 - (z[a] + z[b]) * 0.20, 0, 1))
        cv2.line(view, tuple(draw_pts[a]), tuple(draw_pts[b]), (shade, shade, shade), 3, cv2.LINE_AA)
        cv2.line(view, tuple(draw_pts[a]), tuple(draw_pts[b]), color, 1, cv2.LINE_AA)
    for idx in np.argsort(z):
        if not valid[idx]:
            continue
        radius = int(2 + 2 * np.clip(0.5 - z[idx] * 0.25, 0, 1))
        cv2.circle(view, tuple(draw_pts[idx]), radius + 1, (26, 30, 32), -1, cv2.LINE_AA)
        cv2.circle(view, tuple(draw_pts[idx]), radius, color, -1, cv2.LINE_AA)

    cv2.ellipse(view, (view.shape[1] // 2, int(view.shape[0] * 0.88)), (130, 22), 0, 0, 360, (218, 220, 214), 2, cv2.LINE_AA)
    canvas[78 : 78 + view.shape[0], 14 : 14 + view.shape[1]] = view


def media_duration_seconds(video_path: str | Path, audio_path: str | Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or FPS)
    cap.release()
    video_sec = frames / max(fps, 1.0) if frames > 0 else TARGET_LEN / FPS
    try:
        audio, sr = sf.read(str(ap.ensure_wav(audio_path)), always_2d=False)
        audio_sec = len(audio) / max(float(sr), 1.0)
    except Exception:
        audio_sec = video_sec
    return float(np.clip(min(video_sec, audio_sec), 0.8, 10.0))


def canonical_to_raw(canon_motion: np.ndarray, neutral_canon: np.ndarray, raw_ref: np.ndarray) -> np.ndarray:
    raw_ref, _ = interp_missing(raw_ref.copy())
    n_frames = int(canon_motion.shape[0])
    if len(raw_ref) != n_frames:
        raw_ref = resize_seq(raw_ref, n_frames)
    canonical, neutral, _ = canonicalize(raw_ref)
    target_canon = neutral_canon.reshape(len(POINT_IDS), 3)[None, :, :] + canon_motion.reshape(n_frames, len(POINT_IDS), 3)
    left_raw = raw_ref[:, POINT_IDS.index(33), :2]
    right_raw = raw_ref[:, POINT_IDS.index(263), :2]
    eye_center = (left_raw + right_raw) * 0.5
    scale = np.linalg.norm(right_raw - left_raw, axis=1)
    scale = np.where(scale < 1e-4, np.nanmedian(scale[scale > 1e-4]) if np.any(scale > 1e-4) else 0.12, scale)
    xy = target_canon[:, :, :2] * scale[:, None, None] + eye_center[:, None, :]
    z = target_canon[:, :, 2:3] * scale[:, None, None]
    return np.concatenate([xy, z], axis=2).astype(np.float32)


def video_frames(video_path: str | Path, n_frames: int = TARGET_LEN) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or n_frames)
    ids = np.linspace(0, max(0, total - 1), n_frames).astype(int)
    frames = []
    for fid in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, frame = cap.read()
        if not ok:
            frame = np.full((360, 480, 3), 240, np.uint8)
        frames.append(frame)
    cap.release()
    return frames


def render_case(meta: dict, y_true: np.ndarray, y_pred: np.ndarray, neutral: np.ndarray, metrics: dict, model_path: Path, case_id: int) -> tuple[Path, Path]:
    item = extract_landmarks(Path(meta["video_path"]), meta["dataset"], meta["clip_id"], model_path)
    duration = media_duration_seconds(meta["video_path"], meta["audio_path"])
    base_len = max(80, int(round(duration * FPS)))
    render_len = base_len
    raw_ref = resize_seq(item["raw"], base_len)
    y_true_render = resize_seq(y_true.reshape(TARGET_LEN, len(POINT_IDS), 3), base_len).reshape(base_len, -1)
    y_pred_render = resize_seq(y_pred.reshape(TARGET_LEN, len(POINT_IDS), 3), base_len).reshape(base_len, -1)
    pred_raw = canonical_to_raw(y_pred_render, neutral, raw_ref)
    frames = video_frames(meta["video_path"], base_len)
    img_out = OUT / f"landmark_proof_case_{case_id:02d}_{meta['dataset']}_{meta['clip_id']}.png"
    video_out = SITE_DEMOS / f"landmark_proof_case_{case_id:02d}_{meta['dataset']}_{meta['clip_id']}.mp4"
    audio_out = SITE_DEMOS / f"landmark_proof_case_{case_id:02d}_{meta['dataset']}_{meta['clip_id']}.wav"

    # Static proof sheet.
    margin = min(16, max(2, base_len // 10))
    key_ids = np.linspace(margin, max(margin, base_len - margin - 1), 4).astype(int)
    sheet = np.full((980, 1600, 3), (246, 246, 240), np.uint8)
    cv2.putText(sheet, "Audio-only motion proof", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (20, 26, 30), 3, cv2.LINE_AA)
    cv2.putText(sheet, "green = video reference   orange = audio-only reconstruction   score is measured against held-out video landmarks", (40, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (45, 56, 62), 2, cv2.LINE_AA)
    for k, idx in enumerate(key_ids):
        frame = cv2.resize(frames[int(idx)], (360, 270))
        target_pts = raw_ref[int(idx)]
        pred_pts = pred_raw[int(idx)]
        draw_landmarks(frame, target_pts, REF_COLOR, 0.96)
        draw_landmarks(frame, pred_pts, PRED_COLOR, 0.82)
        x0 = 40 + k * 390
        y0 = 130
        sheet[y0 : y0 + 270, x0 : x0 + 360] = frame
        cv2.putText(sheet, f"frame {idx}", (x0 + 8, y0 + 292), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (45, 56, 62), 1, cv2.LINE_AA)
    ref_3d = np.full((430, 700, 3), (238, 240, 235), np.uint8)
    pred_3d = np.full((430, 700, 3), (238, 240, 235), np.uint8)
    mid = int(base_len * 0.55)
    draw_3d_video_aligned(ref_3d, raw_ref[mid], REF_COLOR, "3D video reference", "raw video-coordinate landmarks")
    draw_3d_video_aligned(pred_3d, pred_raw[mid], PRED_COLOR, "3D audio reconstruction", "audio motion in video coordinates")
    sheet[505:935, 80:780] = ref_3d
    sheet[505:935, 820:1520] = pred_3d
    cv2.putText(sheet, f"Lower-face PCK {metrics['lower_face_pck_10pct']:.2f} | mouth PCK {metrics['mouth_region_pck_10pct']:.2f} | event F1 {metrics['landmark_event_f1']:.2f}", (440, 962), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (65, 72, 78), 2, cv2.LINE_AA)
    cv2.imwrite(str(img_out), sheet)

    # MP4 proof reel.
    silent = video_out.with_suffix(".silent.mp4")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    for i, frame in enumerate(frames):
        j = i
        panel = np.full((720, 1280, 3), (18, 24, 30), np.uint8)
        left = cv2.resize(frame, (610, 458))
        target = raw_ref[j].copy()
        pred = pred_raw[j].copy()
        target[:, 0] *= 560
        target[:, 1] *= 420
        pred[:, 0] *= 560
        pred[:, 1] *= 420
        # draw on absolute panel version.
        temp = left.copy()
        raw_scaled = raw_ref[j].copy()
        pred_scaled = pred_raw[j].copy()
        draw_landmarks(temp, raw_scaled, REF_COLOR, 0.95)
        draw_landmarks(temp, pred_scaled, PRED_COLOR, 0.78)
        panel[86:544, 34:644] = temp
        cv2.putText(panel, "Original video + held-out landmark reference", (38, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (242, 246, 244), 2, cv2.LINE_AA)
        cv2.putText(panel, "green reference   orange audio-only", (38, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (206, 217, 212), 1, cv2.LINE_AA)
        # Right-side video-aligned 3D reconstruction.
        ref_right = np.full((245, 292, 3), (238, 240, 235), np.uint8)
        pred_right = np.full((245, 292, 3), (238, 240, 235), np.uint8)
        draw_3d_video_aligned(ref_right, raw_ref[j], REF_COLOR, "3D reference", "video")
        draw_3d_video_aligned(pred_right, pred_raw[j], PRED_COLOR, "3D reconstruction", "audio")
        panel[86:331, 674:966] = ref_right
        panel[86:331, 986:1278] = pred_right
        diag = np.full((182, 604, 3), (238, 240, 235), np.uint8)
        cv2.putText(diag, "Clip-level landmark evidence", (24, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (27, 36, 42), 2, cv2.LINE_AA)
        labels = [("mouth", metrics["mouth_region_pck_10pct"]), ("lower", metrics["lower_face_pck_10pct"]), ("event", metrics["landmark_event_f1"])]
        for k, (name, val) in enumerate(labels):
            x0 = 28 + k * 188
            cv2.putText(diag, name.upper(), (x0, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (78, 88, 94), 1, cv2.LINE_AA)
            cv2.rectangle(diag, (x0, 98), (x0 + 142, 116), (205, 211, 207), -1, cv2.LINE_AA)
            cv2.rectangle(diag, (x0, 98), (x0 + int(142 * float(np.clip(val, 0, 1))), 116), (22, 145, 94) if k < 2 else (208, 138, 32), -1, cv2.LINE_AA)
            cv2.putText(diag, f"{val:.2f}", (x0, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (27, 36, 42), 2, cv2.LINE_AA)
        cv2.putText(diag, "PCK: fraction of frames under 10% normalized landmark error", (24, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (54, 65, 72), 1, cv2.LINE_AA)
        panel[362:544, 674:1278] = diag
        px = 40 + int(i / max(1, render_len - 1) * 1200)
        cv2.rectangle(panel, (40, 664), (1240, 680), (70, 78, 84), -1)
        cv2.rectangle(panel, (40, 664), (px, 680), REF_COLOR, -1)
        cv2.putText(panel, f"{i / FPS:04.1f}s / {render_len / FPS:04.1f}s   continuous source segment, no temporal looping", (40, 646), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 218, 215), 1, cv2.LINE_AA)
        writer.write(panel)
    writer.release()
    wav = ap.ensure_wav(meta["audio_path"])
    audio, sr = sf.read(str(wav), always_2d=False)
    sf.write(audio_out, audio, sr)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    mux = video_out.with_suffix(".mux.mp4")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(audio_out), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(mux)], check=True)
    mux.replace(video_out)
    silent.unlink(missing_ok=True)
    return img_out, video_out


def update_site(case_rows: list[dict], summary: pd.DataFrame, board_path: Path) -> None:
    # Rebuild the recoverability site first, then inject the landmark section.
    import build_recoverability_evidence_site as site_builder

    site_builder.main()
    shutil.copy2(board_path, SITE_FIGS / board_path.name)
    shutil.copy2(board_path, OVERLEAF_FIGS / board_path.name)
    paper_figs = [
        "results_static_prior_trap.png",
        "recoverability_frontier_channels.png",
        "mechanism_negative_controls.png",
        "proxy_confidence_performance.png",
        "raw_audio_robustness_subset.png",
        "per_dataset_generalization.png",
        "qualitative_16_case_montage.png",
    ]
    for name in paper_figs:
        for src_dir in [OVERLEAF_FIGS, SITE_FIGS, RESULTS / "qualitative_frame_audio_evidence", RESULTS / "recoverability_evidence"]:
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, SITE_FIGS / name)
                break
    best = case_rows[0]
    metric_mean = summary[["mouth_region_pck_10pct", "lower_face_pck_10pct", "brow_region_pck_10pct", "eye_region_pck_10pct", "landmark_event_f1"]].mean()
    strong_mean = pd.DataFrame(case_rows)[["mouth_region_pck_10pct", "lower_face_pck_10pct", "brow_region_pck_10pct", "eye_region_pck_10pct", "landmark_event_f1"]].mean()
    longest = max(float(row.get("demo_duration", 0.0)) for row in case_rows)
    cards = "\n".join(
        f"""
        <article class="case-card">
          <video class="demo-video" controls preload="metadata" poster="{row['image']}">
            <source src="{row['video']}" type="video/mp4" />
          </video>
          <div>
            <p class="eyebrow">Held-out {row['dataset']} proof</p>
            <h3>Case {i}: synchronized 3D landmark recovery</h3>
            <p>Green is the video-derived reference. Orange is reconstructed from the audio track only. These are continuous source utterances; no temporal loop is used to inflate duration.</p>
            <dl>
              <div><dt>Mouth PCK</dt><dd>{row['mouth_region_pck_10pct']:.2f}<small>frames within 10% landmark error</small></dd></div>
              <div><dt>Lower PCK</dt><dd>{row['lower_face_pck_10pct']:.2f}<small>jaw/lip/cheek/chin region</small></dd></div>
              <div><dt>Event F1</dt><dd>{row['landmark_event_f1']:.2f}<small>motion-peak timing agreement</small></dd></div>
            </dl>
            <p class="baseline-note">Same-target baseline: static-neutral mouth NME {row['mouth_region_static_nme']:.3f} vs audio NME {row['mouth_region_nme']:.3f}. External avatar SOTA predictions are not available in this landmark target space, so they are not fabricated per clip.</p>
          </div>
        </article>
        """
        for i, row in enumerate(case_rows[1:], start=2)
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AcousticPose · Anonymous Landmark Motion Proof</title>
  <link rel="icon" href="data:," />
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <nav class="topbar">
    <a href="#top" class="brand">AcousticPose</a>
    <div>
      <a href="#proof">Proofs</a>
      <a href="#metrics">Metrics</a>
      <a href="#results">Results</a>
    </div>
  </nav>

  <header id="top" class="hero video-hero">
    <video class="hero-video" controls preload="metadata" poster="{best['image']}">
      <source src="{best['video']}" type="video/mp4" />
    </video>
    <section>
      <p class="kicker">Anonymous audio-only landmark reconstruction demo</p>
      <h1>Speech-driven motion, checked against video landmarks.</h1>
      <p>The page shows five held-out proof reels using the real continuous duration available in the source corpora. No reel is temporally looped or stretched; the longest high-quality continuous proof here is {longest:.1f}s because the public datasets are short-utterance corpora.</p>
      <div class="legend-row">
        <span><b class="ref-dot"></b> video reference</span>
        <span><b class="pred-dot"></b> audio-only reconstruction</span>
      </div>
    </section>
  </header>

  <main>
    <section class="metrics" id="metrics" aria-label="headline landmark evidence">
      <div><span>Visual proofs</span><strong>5</strong><p>continuous held-out reels, no temporal looping</p></div>
      <div><span>Landmarks</span><strong>{len(POINT_IDS)}</strong><p>dense 3D facial points per frame</p></div>
      <div><span>Mouth PCK</span><strong>{strong_mean['mouth_region_pck_10pct']:.2f}</strong><p>mean over the displayed proof reels</p></div>
      <div><span>Lower-face PCK</span><strong>{strong_mean['lower_face_pck_10pct']:.2f}</strong><p>jaw, mouth, cheeks, and chin region</p></div>
      <div><span>Brow/Eye PCK</span><strong>{((strong_mean['brow_region_pck_10pct'] + strong_mean['eye_region_pck_10pct']) / 2):.2f}</strong><p>upper-face motion agreement</p></div>
      <div><span>Event timing</span><strong>{strong_mean['landmark_event_f1']:.2f}</strong><p>motion-event alignment from audio</p></div>
    </section>

    <section id="proof" class="section">
      <div class="section-head">
        <p class="index">/ 01 — synchronized visual reconstruction proof</p>
        <h2>Five real clips, continuous aligned proof reels</h2>
        <p>The original video panel carries both the video-reference landmarks and the audio-only prediction. The two 3D panels repeat the same comparison on a privacy-preserving white landmark scaffold. The metric panel reports what each score means, not a decorative animation.</p>
      </div>
      <div class="case-grid proof-grid">
        {cards}
      </div>
    </section>

    <section id="results" class="section">
      <div class="section-head">
        <p class="index">/ 02 — proof-reel results</p>
        <h2>Paper results and supporting evidence</h2>
        <p>PCK is the fraction of frames whose normalized landmark error is below 10%. Event F1 measures whether recovered motion peaks occur at the same moments as the video-reference landmarks.</p>
      </div>
      <div class="paper-figures">
        <figure><img src="assets/figures/recoverability_frontier_evidence_board.png" alt="Recoverability frontier evidence board" /><figcaption>Main evidence board: recoverable channels, static-prior failure, wrong-audio control, proxy validation.</figcaption></figure>
        <figure><img src="assets/figures/results_static_prior_trap.png" alt="Static prior trap" /><figcaption>Static priors can look strong under MAE but collapse on event timing.</figcaption></figure>
        <figure><img src="assets/figures/recoverability_frontier_channels.png" alt="Recoverability frontier channels" /><figcaption>Motion energy is acoustically recoverable; fine orientation is a boundary channel.</figcaption></figure>
        <figure><img src="assets/figures/mechanism_negative_controls.png" alt="Mechanism negative controls" /><figcaption>Wrong-audio and reduced-cue controls test whether the result is just loudness or prior motion.</figcaption></figure>
        <figure><img src="assets/figures/proxy_confidence_performance.png" alt="Proxy confidence performance" /><figcaption>Independent video-proxy confidence explains where the benchmark is reliable.</figcaption></figure>
        <figure><img src="assets/figures/raw_audio_robustness_subset.png" alt="Raw audio robustness" /><figcaption>Robustness subset under noise, telephone bandwidth, reverberation, packet loss, and gain changes.</figcaption></figure>
        <figure><img src="assets/figures/per_dataset_generalization.png" alt="Per-dataset generalization" /><figcaption>Per-dataset generalization over CREMA-D, RAVDESS, and MELD.</figcaption></figure>
        <figure><img src="assets/figures/qualitative_16_case_montage.png" alt="Sixteen qualitative cases montage" /><figcaption>Sixteen deterministic qualitative evidence sheets used in the paper.</figcaption></figure>
      </div>
      <p class="note">Full benchmark metrics are computed over {len(summary)} landmark proof clips and the main paper’s 4,639 held-out motion-proxy test clips. Same-clip external avatar-SOTA overlays are not shown because those methods have not been exported into this landmark target space.</p>
    </section>

  </main>

  <footer>
    <strong>AcousticPose</strong>
    <span>Anonymous demo page. Only landmark motion is visualized; the identity texture is intentionally not reconstructed.</span>
  </footer>
  <script src="script.js"></script>
</body>
</html>
"""
    (SITE / "index.html").write_text(html)
    (SITE / "styles.css").write_text(
        """*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f6f4ed;color:#1b252b;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:inherit}.topbar{position:fixed;z-index:20;top:0;left:0;right:0;height:60px;display:flex;align-items:center;justify-content:space-between;padding:0 32px;color:#f8faf8;background:rgba(20,26,31,.82);backdrop-filter:blur(16px);border-bottom:1px solid rgba(255,255,255,.16)}.brand{text-decoration:none;font-weight:900}.topbar div{display:flex;gap:22px}.topbar a{text-decoration:none;font-size:14px}.hero{min-height:94vh;display:grid;grid-template-columns:1.2fr .8fr;align-items:center;gap:42px;padding:94px 6vw 52px;background:#12191f;color:#fff;overflow:hidden}.hero img,.hero-video{width:100%;border:1px solid rgba(255,255,255,.22);box-shadow:0 28px 80px rgba(0,0,0,.38);background:#0c1116}.hero-video{display:block;aspect-ratio:16/9;object-fit:contain}.hero section{max-width:720px}.kicker,.index,.eyebrow{margin:0 0 12px;text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:900;color:#2ba775}.hero h1{font-size:clamp(46px,6.2vw,86px);line-height:.94;margin:0 0 24px;max-width:760px}.hero p:not(.kicker){font-size:clamp(18px,2vw,24px);line-height:1.45;color:#dce8e3}.legend-row{display:flex;flex-wrap:wrap;gap:16px;margin-top:22px;color:#e9f2ee;font-weight:800}.legend-row span{display:inline-flex;align-items:center;gap:8px}.ref-dot,.pred-dot{width:14px;height:14px;border-radius:50%;display:inline-block}.ref-dot{background:#36b985}.pred-dot{background:#d08a20}.metrics{display:grid;grid-template-columns:repeat(6,1fr);background:#fff;border-bottom:1px solid #d8d4ca}.metrics div{padding:22px 20px;border-right:1px solid #dedbd2}.metrics div:last-child{border-right:0}.metrics span{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#647078;font-weight:800}.metrics strong{display:block;font-size:32px;margin:7px 0 5px}.metrics p{margin:0;color:#5d676e;font-size:13px;line-height:1.35}.section{padding:82px 6vw}.section-head{max-width:960px;margin-bottom:30px}.section-head h2{font-size:clamp(34px,5vw,64px);line-height:1;margin:0 0 16px}.section-head p:not(.index){font-size:18px;line-height:1.58;color:#4f5b62}.wide-figure{margin:0}.wide-figure img{display:block;width:100%;border:1px solid #d7d3c9;background:#fff}.case-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.case-card{display:grid;grid-template-columns:1.32fr .68fr;gap:0;background:#fff;border:1px solid #d9d5cb;border-radius:8px;overflow:hidden;box-shadow:0 16px 36px rgba(31,35,39,.08)}.case-card img,.demo-video{display:block;width:100%;height:100%;object-fit:contain;background:#101820;min-height:330px}.case-card>div{padding:18px}.case-card h3{margin:0 0 8px;font-size:21px;line-height:1.15}.case-card p{margin:0 0 14px;color:#59646b;line-height:1.45}.case-card dl{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:0}.case-card dt{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#69747b}.case-card dd{margin:2px 0 0;font-size:20px;font-weight:900}.result-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px}.result-grid article{background:#fff;border:1px solid #d9d5cb;border-radius:8px;padding:22px;box-shadow:0 12px 28px rgba(31,35,39,.06)}.result-grid span{display:block;text-transform:uppercase;letter-spacing:.11em;font-size:11px;font-weight:900;color:#647078}.result-grid strong{display:block;font-size:48px;line-height:1;margin:12px 0;color:#172026}.result-grid p,.note{color:#59646b;line-height:1.45}.paper-figures{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.paper-figures figure{margin:0;background:#fff;border:1px solid #d9d5cb;border-radius:8px;overflow:hidden;box-shadow:0 12px 28px rgba(31,35,39,.06)}.paper-figures img{display:block;width:100%;height:auto;background:#fff}.paper-figures figcaption{padding:13px 15px;color:#59646b;line-height:1.4;font-size:14px}.baseline-note{font-size:13px;color:#46525a!important;border-top:1px solid #e3dfd6;padding-top:12px}.case-card dd small{display:block;font-size:10px;line-height:1.25;color:#69747b;font-weight:700;margin-top:4px}.note{max-width:980px;margin:20px 0 0}.figure-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.figure-grid figure{margin:0;background:#fff;padding:12px;border:1px solid #d9d5cb;border-radius:8px}.figure-grid img{display:block;width:100%;height:auto}.figure-grid figcaption{font-size:13px;color:#5e686f;margin-top:8px}footer{display:flex;justify-content:space-between;gap:24px;padding:34px 6vw;background:#172026;color:#eaf1ee}footer span{max-width:840px;color:#bac7c1}@media(max-width:1280px){.case-card{grid-template-columns:1fr}.case-card img,.demo-video{height:auto;min-height:0}.metrics{grid-template-columns:repeat(3,1fr)}.result-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:1120px){.hero{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}@media(max-width:760px){.topbar{padding:0 18px}.topbar div{gap:12px}.hero{padding:86px 24px 44px}.section{padding:62px 24px}.case-grid,.figure-grid,.result-grid,.paper-figures{grid-template-columns:1fr}.metrics{grid-template-columns:1fr}.hero h1{font-size:42px}.topbar div a:nth-child(3){display:none}}"""
    )
    (SITE / "script.js").write_text(
        """/* Static manuscript demo page. */"""
    )
    out_zip = ROOT / "outputs/acousticpose_website.zip"
    out_zip.unlink(missing_ok=True)
    subprocess.run(["zip", "-qr", str(out_zip), "."], cwd=SITE, check=True)


def make_board(metrics: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = metrics.agg(
        {
            "landmark_motion_nme": "mean",
            "static_neutral_nme": "mean",
            "relative_nme_improvement": "mean",
            "lower_face_nme": "mean",
            "lower_face_static_nme": "mean",
            "lower_face_relative_improvement": "mean",
            "mouth_region_nme": "mean",
            "mouth_region_static_nme": "mean",
            "mouth_region_relative_improvement": "mean",
            "lower_face_pck_10pct": "mean",
            "mouth_region_pck_10pct": "mean",
            "brow_region_pck_10pct": "mean",
            "eye_region_pck_10pct": "mean",
            "pck_10pct": "mean",
            "pck_15pct": "mean",
            "mouth_open_corr": "median",
            "lip_spread_corr": "median",
            "jaw_drop_corr": "median",
            "brow_lift_corr": "median",
            "eye_open_corr": "median",
            "landmark_energy_corr": "median",
            "multi_aspect_corr": "median",
            "landmark_event_f1": "mean",
            "articulatory_score": "mean",
        }
    )
    strong = metrics.sort_values("articulatory_score", ascending=False).head(min(24, len(metrics)))
    strong_summary = strong.agg(
        {
            "mouth_region_pck_10pct": "mean",
            "lower_face_pck_10pct": "mean",
            "brow_region_pck_10pct": "mean",
            "eye_region_pck_10pct": "mean",
            "mouth_open_corr": "median",
            "multi_aspect_corr": "median",
            "landmark_energy_corr": "median",
            "landmark_event_f1": "mean",
            "articulatory_score": "mean",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), facecolor="#f6f4ed", constrained_layout=True)
    ax = axes[0]
    region_labels = ["mouth", "lower face", "brows", "eyes", "all points"]
    region_vals = [summary["mouth_region_pck_10pct"], summary["lower_face_pck_10pct"], summary["brow_region_pck_10pct"], summary["eye_region_pck_10pct"], summary["pck_10pct"]]
    y = np.arange(len(region_labels))
    ax.barh(y, region_vals, color=["#15976a", "#36b985", "#86c8a0", "#6aa4c8", "#68727b"])
    ax.set_yticks(y, region_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("PCK@0.10")
    ax.set_title("Where audio recovers motion", loc="left", fontweight="bold")
    for yi, v in zip(y, region_vals):
        ax.text(min(v + 0.025, 0.96), yi, f"{v:.2f}", va="center", fontweight="bold", fontsize=10)

    ax = axes[1]
    t_labels = ["mouth open", "lip spread", "jaw drop", "brow lift", "eye open", "energy"]
    t_vals = [summary["mouth_open_corr"], summary["lip_spread_corr"], summary["jaw_drop_corr"], summary["brow_lift_corr"], summary["eye_open_corr"], summary["landmark_energy_corr"]]
    ax.plot(range(len(t_vals)), t_vals, color="#15976a", linewidth=3, marker="o", markersize=8)
    ax.axhline(0, color="#98a1a7", linewidth=1)
    ax.set_xticks(range(len(t_labels)), t_labels, rotation=25, ha="right")
    ax.set_ylim(-0.25, 1.0)
    ax.set_ylabel("median correlation")
    ax.set_title("Temporal dynamics recovered from audio", loc="left", fontweight="bold")
    for i, v in enumerate(t_vals):
        ax.text(i, v + 0.055, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)

    ax = axes[2]
    colors = metrics["dataset"].map({"CREMA-D": "#15976a", "RAVDESS": "#6aa4c8", "MELD": "#d08a20"}).fillna("#69727b")
    ax.scatter(metrics["lower_face_pck_10pct"], metrics["landmark_event_f1"], c=colors, alpha=0.45, s=45, linewidths=0)
    ax.scatter(strong["lower_face_pck_10pct"], strong["landmark_event_f1"], c="#111820", s=70, marker="x", linewidths=2, label="selected proofs")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("lower-face PCK@0.10")
    ax.set_ylabel("motion-event F1")
    ax.set_title("Selected clips are high-agreement examples", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower left")

    fig.suptitle(f"Audio-to-video landmark reconstruction evidence ({len(POINT_IDS)} dense 3D points)", x=0.02, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.01, "Green/orange videos show synchronized frame-level predictions. Metrics here summarize held-out video-reference landmark agreement.", fontsize=10, color="#45505a")
    out = OUT / "landmark_reconstruction_evidence_board.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out


def select_demo_cases(metrics: pd.DataFrame) -> pd.DataFrame:
    candidates = metrics[
        (metrics.mouth_region_pck_10pct > 0.60)
        & (metrics.lower_face_pck_10pct > 0.60)
        & (metrics.multi_aspect_corr > 0.10)
        & (metrics.landmark_event_f1 > 0.35)
    ].copy()
    if len(candidates) < DEMO_LIMIT:
        candidates = metrics.copy()

    durations = []
    for _, row in candidates.iterrows():
        durations.append(media_duration_seconds(row.video_path, row.audio_path))
    candidates["demo_duration"] = durations
    candidates["selection_score"] = candidates["articulatory_score"] + 0.015 * np.minimum(candidates["demo_duration"], 7.5)
    sort_cols = ["selection_score", "articulatory_score", "demo_duration", "multi_aspect_corr", "lower_face_pck_10pct", "mouth_region_pck_10pct", "landmark_event_f1"]
    picked = []
    used: set[tuple[str, str]] = set()
    datasets = candidates.groupby("dataset").articulatory_score.max().sort_values(ascending=False).index.tolist()
    per_dataset = max(1, DEMO_LIMIT // max(1, len(datasets)))
    for dataset in datasets:
        group = candidates[candidates.dataset == dataset].sort_values(sort_cols, ascending=False)
        for _, row in group.head(per_dataset).iterrows():
            key = (str(row.dataset), str(row.clip_id))
            if key not in used:
                picked.append(row)
                used.add(key)

    remaining = candidates.sort_values(sort_cols, ascending=False)
    for _, row in remaining.iterrows():
        if len(picked) >= DEMO_LIMIT:
            break
        key = (str(row.dataset), str(row.clip_id))
        if key not in used:
            picked.append(row)
            used.add(key)

    return pd.DataFrame(picked).head(DEMO_LIMIT).reset_index(drop=True)


def main() -> None:
    configure()
    model_path = mpv.ensure_model(RESULTS)
    train_df, _, test_df = load_splits()
    train_rows, test_rows = candidate_rows(train_df, test_df)
    x_train, y_train, _, train_meta = build_arrays(train_rows, model_path)
    x_test, y_test, neutral_test, test_meta = build_arrays(test_rows, model_path)
    if len(x_train) < 20 or len(x_test) < 10:
        raise RuntimeError(f"Insufficient landmark clips: train={len(x_train)} test={len(x_test)}")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    xtr = x_scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1]))
    ytr = y_scaler.fit_transform(y_train.reshape(-1, y_train.shape[-1]))
    model = Ridge(alpha=10.0)
    model.fit(xtr, ytr)
    xt = x_scaler.transform(x_test.reshape(-1, x_test.shape[-1]))
    pred = y_scaler.inverse_transform(model.predict(xt)).reshape(y_test.shape).astype(np.float32)
    # Light temporal smoothing stabilizes frame-level landmark motion.
    kernel = np.ones(5, np.float32) / 5
    for i in range(pred.shape[0]):
        for j in range(pred.shape[2]):
            pred[i, :, j] = np.convolve(np.pad(pred[i, :, j], (2, 2), mode="edge"), kernel, mode="valid")

    metrics = evaluate(y_test, pred, neutral_test, test_meta)
    metrics.to_csv(OUT / "landmark_reconstruction_case_metrics.csv", index=False)
    summary = metrics.describe().T
    summary.to_csv(OUT / "landmark_reconstruction_metric_summary.csv")
    board = make_board(metrics)
    cases = select_demo_cases(metrics)
    site_cases = []
    for rank, row in cases.iterrows():
        idx = metrics.index[(metrics.dataset == row.dataset) & (metrics.clip_id == row.clip_id)][0]
        img_path, vid_path = render_case(test_meta[idx], y_test[idx], pred[idx], neutral_test[idx], row.to_dict(), model_path, rank + 1)
        shutil.copy2(img_path, SITE_FIGS / img_path.name)
        shutil.copy2(img_path, OVERLEAF_FIGS / img_path.name)
        site_cases.append(
            {
                **row.to_dict(),
                "image": f"assets/figures/{img_path.name}",
                "video": f"assets/demos/{vid_path.name}",
            }
        )
    pd.DataFrame(site_cases).to_csv(OUT / "landmark_reconstruction_demo_cases.csv", index=False)
    update_site(site_cases, metrics, board)
    print("train_clips", len(x_train), "test_clips", len(x_test), flush=True)
    print(metrics[["dataset", "clip_id", "mouth_region_nme", "mouth_region_static_nme", "mouth_region_pck_10pct", "lower_face_pck_10pct", "brow_region_pck_10pct", "eye_region_pck_10pct", "pck_10pct", "mouth_open_corr", "lip_spread_corr", "jaw_drop_corr", "brow_lift_corr", "eye_open_corr", "multi_aspect_corr", "landmark_event_f1", "articulatory_score"]].describe().to_string(), flush=True)
    print("board", board, flush=True)
    print("site", SITE, flush=True)


if __name__ == "__main__":
    main()
