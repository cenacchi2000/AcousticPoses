#!/usr/bin/env python3
"""Local real-data AcousticPose pipeline.

This is a local, restartable version of the Colab notebook. It deliberately
keeps strict gates for paper claims: quick or small runs are useful for testing,
but they are not treated as AAAI-ready evidence.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shutil
import subprocess
import tarfile
import time
import urllib.request
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover - optional dependency
    mp = None
    print("MediaPipe import failed; optical-flow targets will be used:", repr(exc))


SEED = 42
TARGET_DIMS = ["head_yaw", "head_pitch", "head_roll", "torso_lean", "motion_energy"]
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
RAVDESS_INTENSITY = {"01": "normal", "02": "strong"}
CREMAD_EMOTIONS = {"ANG": "angry", "DIS": "disgust", "FEA": "fearful", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
CREMAD_LEVELS = {"LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified"}


@dataclass
class Config:
    project_root: Path
    data_root: Path
    cache_root: Path
    output_root: Path
    sota_root: Path
    sr: int = 16000
    fps: int = 25
    target_len: int = 160
    max_clip_sec: float = 12.0
    batch_size: int = 16
    epochs: int = 80
    patience: int = 12
    lr: float = 2e-4
    wd: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    strict: bool = True
    min_clips: int = 500
    min_datasets: int = 3
    min_rel_impr: float = 0.15
    min_ablation_gain: float = 0.08
    target_backend: str = "auto"
    hidden_size: int = 256

    @property
    def fig_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def table_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def model_dir(self) -> Path:
        return self.output_root / "models"

    def mkdirs(self) -> None:
        for path in [self.project_root, self.data_root, self.cache_root, self.output_root, self.fig_dir, self.table_dir, self.model_dir, self.sota_root]:
            path.mkdir(parents=True, exist_ok=True)


CFG: Config
MP_FACE = None
MP_POSE = None


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sh(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(">>", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=check)


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required. Install it with Homebrew, e.g. `brew install ffmpeg`.")


def download_url(url: str, dst: Path, chunk: int = 1024 * 1024) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print("Already exists", dst)
        return dst
    print("Downloading", url, "->", dst)
    with urllib.request.urlopen(url) as response, open(dst, "wb") as handle:
        total = int(response.headers.get("Content-Length", 0) or 0)
        done = 0
        while True:
            block = response.read(chunk)
            if not block:
                break
            handle.write(block)
            done += len(block)
            if total:
                print(f"{done / total:6.1%}", end="\r")
    print("\nDone")
    return dst


def extract_archive(archive: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if marker.exists():
        print("Already extracted", out_dir)
        return out_dir
    print("Extracting", archive.name, "->", out_dir)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zip_file:
            zip_file.extractall(out_dir)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(out_dir)
    else:
        raise ValueError(f"Unknown archive type: {archive}")
    marker.write_text(time.ctime())
    return out_dir


def extract_nested_archives(root: Path) -> None:
    """Extract archives found inside an already extracted dataset directory."""
    root = Path(root)
    for archive in sorted(root.rglob("*.tar.gz")) + sorted(root.rglob("*.tgz")) + sorted(root.rglob("*.zip")):
        if archive.name == "MELD.Raw.tar.gz":
            continue
        out_dir = archive.with_suffix("").with_suffix("") if archive.name.endswith(".tar.gz") else archive.with_suffix("")
        marker = out_dir / ".extracted"
        if marker.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print("Extracting nested", archive.name, "->", out_dir)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as zip_file:
                zip_file.extractall(out_dir)
        else:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(out_dir)
        marker.write_text(time.ctime())


def download_ravdess_actor(actor: int) -> Path:
    actor_s = f"{actor:02d}"
    fname = f"Video_Speech_Actor_{actor_s}.zip"
    url = f"https://zenodo.org/records/1188976/files/{fname}?download=1"
    archive = CFG.data_root / "RAVDESS" / fname
    download_url(url, archive)
    return extract_archive(archive, CFG.data_root / "RAVDESS" / f"Actor_{actor_s}")


def download_ravdess(actors: list[int]) -> None:
    for actor in actors:
        download_ravdess_actor(actor)


def clone_cremad() -> Path:
    target = CFG.data_root / "CREMA-D"
    if target.exists():
        print("CREMA-D already exists", target)
        return target
    if shutil.which("git-lfs") is None:
        raise RuntimeError("git-lfs is required for CREMA-D. Install it, then run `git lfs install`.")
    sh(["git", "lfs", "clone", "https://github.com/CheyneyComputerScience/CREMA-D.git", str(target)])
    return target


def download_meld_raw() -> Path:
    url = "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
    archive = CFG.data_root / "MELD" / "MELD.Raw.tar.gz"
    download_url(url, archive)
    out_dir = extract_archive(archive, CFG.data_root / "MELD" / "raw")
    extract_nested_archives(out_dir)
    return out_dir


def parse_ravdess(path: Path) -> dict:
    parts = path.stem.split("-")
    out = {"clip_id": path.stem, "dataset": "RAVDESS"}
    if len(parts) == 7:
        _, _, emotion, intensity, _, _, actor = parts
        out.update(
            label_emotion=RAVDESS_EMOTIONS.get(emotion, "unknown"),
            label_intensity=RAVDESS_INTENSITY.get(intensity, "unknown"),
            speaker_id=f"ravdess_actor_{actor}",
            gender="male" if int(actor) % 2 == 1 else "female",
        )
    return out


def parse_cremad(path: Path) -> dict:
    parts = path.stem.split("_")
    out = {"clip_id": path.stem, "dataset": "CREMA-D"}
    if len(parts) >= 4:
        speaker, _, emotion, level = parts[:4]
        out.update(
            speaker_id=f"cremad_{speaker}",
            label_emotion=CREMAD_EMOTIONS.get(emotion, emotion),
            label_intensity=CREMAD_LEVELS.get(level, level),
        )
    return out


def index_ravdess(root: Path) -> pd.DataFrame:
    rows = []
    for video in root.rglob("*.mp4"):
        row = parse_ravdess(video)
        row.update(video_path=str(video), audio_path=str(video))
        rows.append(row)
    return pd.DataFrame(rows)


def index_cremad(root: Path) -> pd.DataFrame:
    videos = list((root / "VideoFlash").rglob("*.flv")) + list(root.rglob("*.mp4"))
    audios = {p.stem: p for p in list((root / "AudioWAV").rglob("*.wav")) + list(root.rglob("*.wav"))}
    rows = []
    for video in videos:
        row = parse_cremad(video)
        row.update(video_path=str(video), audio_path=str(audios.get(video.stem, video)))
        rows.append(row)
    return pd.DataFrame(rows)


def index_meld(root: Path) -> pd.DataFrame:
    rows = []
    videos_by_name: dict[str, list[Path]] = {}
    for path in root.rglob("*.mp4"):
        videos_by_name.setdefault(path.name, []).append(path)

    def split_for_csv(csv_path: Path) -> str:
        name = csv_path.name.lower()
        parent = str(csv_path.parent).lower()
        if "train" in name or "/train" in parent:
            return "train"
        if "dev" in name or "/dev" in parent:
            return "dev"
        return "test"

    def choose_meld_video(vid: str, split: str) -> Optional[Path]:
        candidates = videos_by_name.get(vid, [])
        if not candidates:
            return None
        preferred = {
            "train": ["train_splits", "/train/"],
            "dev": ["dev_splits_complete", "/dev/"],
            "test": ["output_repeated_splits_test", "/test/"],
        }[split]
        for marker in preferred:
            for candidate in candidates:
                normalized = str(candidate).replace("\\", "/").lower()
                if marker in normalized:
                    return candidate
        return candidates[0]

    for csv_path in root.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if {"Dialogue_ID", "Utterance_ID"}.issubset(df.columns):
            split = split_for_csv(csv_path)
            for _, item in df.iterrows():
                vid = f"dia{int(item['Dialogue_ID'])}_utt{int(item['Utterance_ID'])}.mp4"
                video_path = choose_meld_video(vid, split)
                if video_path is None:
                    continue
                rows.append(
                    {
                        "clip_id": vid[:-4],
                        "dataset": "MELD",
                        "video_path": str(video_path),
                        "audio_path": str(video_path),
                        "label_emotion": item.get("Emotion"),
                        "speaker_id": f"meld_{item.get('Speaker', 'unk')}",
                        "split": split,
                    }
                )
    return pd.DataFrame(rows)


def index_motion_root(root: Optional[Path], name: str) -> pd.DataFrame:
    if root is None or not root.exists():
        return pd.DataFrame()
    motions = {}
    for pattern in ["*.npy", "*.npz", "*.json"]:
        for path in root.rglob(pattern):
            motions[path.stem] = path
    rows = []
    audio_files = []
    for pattern in ["*.wav", "*.mp3", "*.flac", "*.m4a"]:
        audio_files.extend(root.rglob(pattern))
    for audio in audio_files:
        motion = motions.get(audio.stem)
        if motion:
            rows.append(
                {
                    "clip_id": audio.stem,
                    "dataset": name,
                    "audio_path": str(audio),
                    "video_path": None,
                    "motion_path": str(motion),
                    "speaker_id": f"{name}_{audio.parent.name}",
                }
            )
    return pd.DataFrame(rows)


def index_av_root(root: Optional[Path], name: str = "LOCAL_AV") -> pd.DataFrame:
    if root is None or not root.exists():
        return pd.DataFrame()
    rows = []
    for video in list(root.rglob("*.mp4")) + list(root.rglob("*.mov")) + list(root.rglob("*.mkv")) + list(root.rglob("*.flv")):
        rows.append(
            {
                "clip_id": video.stem,
                "dataset": name,
                "video_path": str(video),
                "audio_path": str(video),
                "speaker_id": f"{name}_{video.stem}",
            }
        )
    return pd.DataFrame(rows)


def build_master_index(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    if (CFG.data_root / "RAVDESS").exists():
        frames.append(index_ravdess(CFG.data_root / "RAVDESS"))
    if (CFG.data_root / "CREMA-D").exists():
        frames.append(index_cremad(CFG.data_root / "CREMA-D"))
    if (CFG.data_root / "MELD").exists():
        frames.append(index_meld(CFG.data_root / "MELD"))
    for root, name in [
        (args.beat_root, "BEAT"),
        (args.beat2_root, "BEAT2"),
        (args.talkshow_root, "TalkSHOW"),
        (args.audio2photoreal_root, "Audio2Photoreal"),
    ]:
        frame = index_motion_root(root, name)
        if len(frame):
            frames.append(frame)
    if args.av_root:
        frame = index_av_root(args.av_root, args.av_dataset_name)
        if len(frame):
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "video_path" not in df:
        df["video_path"] = None
    if "motion_path" not in df:
        df["motion_path"] = None
    df["has_video"] = df["video_path"].map(lambda x: isinstance(x, str) and Path(x).exists())
    df["has_motion_file"] = df["motion_path"].map(lambda x: isinstance(x, str) and Path(x).exists())
    df = df[df.has_video | df.has_motion_file].drop_duplicates(["dataset", "clip_id"]).reset_index(drop=True)
    return df


def ensure_wav(src: str | Path) -> Path:
    src = Path(src)
    if src.suffix.lower() == ".wav":
        return src
    cache = CFG.cache_root / "wav"
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"{hashlib.md5(str(src).encode()).hexdigest()}_{src.stem}.wav"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(CFG.sr), str(dst)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return dst


def resize_seq(arr: np.ndarray, target_len: Optional[int] = None) -> np.ndarray:
    target_len = target_len or CFG.target_len
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if len(arr) == target_len:
        return arr.astype(np.float32)
    if len(arr) == 0:
        width = arr.shape[1] if arr.ndim > 1 else 1
        return np.zeros((target_len, width), np.float32)
    old = np.linspace(0, 1, len(arr))
    new = np.linspace(0, 1, target_len)
    return np.stack([np.interp(new, old, arr[:, dim]) for dim in range(arr.shape[1])], axis=-1).astype(np.float32)


def acoustic_features(audio_path: Path) -> tuple[np.ndarray, list[str]]:
    y, sr = librosa.load(str(audio_path), sr=CFG.sr, mono=True, duration=CFG.max_clip_sec)
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / (np.max(np.abs(y)) + 1e-8)
    hop = max(1, int(sr / CFG.fps))
    n_fft = 1024
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) + 1e-8
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low = spec[freqs < 1000].mean(0)
    mid = spec[(freqs >= 1000) & (freqs < 4000)].mean(0)
    high = spec[freqs >= 4000].mean(0)
    hf_ratio = np.log(high / (low + mid + 1e-8) + 1e-8)
    mid_high = np.log(high / (mid + 1e-8) + 1e-8)
    centroid = librosa.feature.spectral_centroid(S=spec, sr=sr)[0] / (sr / 2)
    roll85 = librosa.feature.spectral_rolloff(S=spec, sr=sr, roll_percent=0.85)[0] / (sr / 2)
    roll95 = librosa.feature.spectral_rolloff(S=spec, sr=sr, roll_percent=0.95)[0] / (sr / 2)
    bandwidth = librosa.feature.spectral_bandwidth(S=spec, sr=sr)[0] / (sr / 2)
    flatness = librosa.feature.spectral_flatness(S=spec)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0]
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    flux = np.r_[0, np.sqrt(np.sum(np.diff(spec, axis=1) ** 2, axis=0))]
    flux = flux / (np.max(flux) + 1e-8)
    rms_d = np.r_[0, np.diff(rms)]
    hf_d = np.r_[0, np.diff(hf_ratio)]
    plosive = np.maximum(0, rms_d) * np.maximum(0, hf_d - np.median(hf_d))
    plosive = plosive / (np.max(plosive) + 1e-8)
    drr = np.log((rms + 1e-6) / (pd.Series(rms).rolling(8, min_periods=1).mean().values + 1e-6))
    off_axis = -np.abs(np.gradient(hf_ratio)) + np.gradient(centroid)
    silence = (rms < np.percentile(rms, 30)).astype(np.float32)
    silence_texture = silence * (flatness + zcr)
    feats = np.stack(
        [
            hf_ratio,
            mid_high,
            centroid,
            roll85,
            roll95,
            bandwidth,
            flatness,
            zcr,
            rms,
            flux,
            plosive,
            drr,
            off_axis,
            silence,
            silence_texture,
            np.gradient(hf_ratio),
            np.gradient(centroid),
            np.gradient(rms),
        ],
        axis=-1,
    )
    names = [
        "hf_ratio",
        "mid_high_ratio",
        "centroid",
        "rolloff85",
        "rolloff95",
        "bandwidth",
        "flatness",
        "zcr",
        "rms",
        "flux",
        "plosive_proxy",
        "drr_proxy",
        "off_axis_proxy",
        "silence_prob",
        "silence_texture",
        "hf_drift",
        "centroid_drift",
        "level_drift",
    ]
    return np.nan_to_num(resize_seq(feats)).astype(np.float32), names


def setup_mediapipe() -> str:
    global MP_FACE, MP_POSE
    if CFG.target_backend == "optical_flow":
        return "forced_optical_flow"
    if mp is None:
        return "mediapipe_not_imported"
    try:
        if hasattr(mp, "solutions"):
            MP_FACE = mp.solutions.face_mesh
            MP_POSE = mp.solutions.pose
            return "mediapipe_solutions_ok"
    except Exception as exc:
        print("MediaPipe solution lookup failed:", repr(exc))
    try:
        import mediapipe.python.solutions.face_mesh as face_mesh
        import mediapipe.python.solutions.pose as pose

        MP_FACE = face_mesh
        MP_POSE = pose
        return "mediapipe_internal_solutions_ok"
    except Exception as exc:
        print("MediaPipe internal import failed; optical flow will be used:", repr(exc))
        return "mediapipe_unavailable_optical_flow_only"


def optical_flow_targets(video: str | Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print("Could not open video for optical flow:", video)
        return np.zeros((CFG.target_len, 5), np.float32)
    fps = cap.get(cv2.CAP_PROP_FPS) or CFG.fps
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or CFG.max_clip_sec * fps
    max_frames = int(min(CFG.max_clip_sec * fps, frame_count))
    prev = None
    rows = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            fx = flow[..., 0]
            fy = flow[..., 1]
            mag, _ = cv2.cartToPolar(fx, fy)
            h, w = gray.shape
            upper = mag[: h // 2, :].mean()
            lower = mag[h // 2 :, :].mean()
            left = mag[:, : w // 2].mean()
            right = mag[:, w // 2 :].mean()
            rows.append(
                [
                    float(fx.mean()),
                    float(fy.mean()),
                    float((right - left) / (right + left + 1e-6)),
                    float((lower - upper) / (lower + upper + 1e-6)),
                    float(mag.mean()),
                ]
            )
        prev = gray
    cap.release()
    if not rows:
        return np.zeros((CFG.target_len, 5), np.float32)
    return resize_seq(np.asarray(rows, np.float32)).astype(np.float32)


def mediapipe_targets(video: str | Path) -> np.ndarray:
    if MP_FACE is None or MP_POSE is None:
        return optical_flow_targets(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return optical_flow_targets(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or CFG.fps
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or CFG.max_clip_sec * fps
    max_frames = int(min(CFG.max_clip_sec * fps, frame_count))
    rows = []
    last = None
    try:
        with MP_FACE.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh, MP_POSE.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            for _ in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2RGB)
                yaw = pitch = roll = torso = np.nan
                face_result = face_mesh.process(rgb)
                if getattr(face_result, "multi_face_landmarks", None):
                    lm = face_result.multi_face_landmarks[0].landmark
                    left_eye = np.array([lm[33].x, lm[33].y], dtype=np.float32)
                    right_eye = np.array([lm[263].x, lm[263].y], dtype=np.float32)
                    nose = np.array([lm[1].x, lm[1].y], dtype=np.float32)
                    mouth = np.array([lm[13].x, lm[13].y], dtype=np.float32)
                    eye = 0.5 * (left_eye + right_eye)
                    eye_vec = right_eye - left_eye
                    scale = float(np.linalg.norm(eye_vec) + 1e-6)
                    roll = float(np.arctan2(eye_vec[1], eye_vec[0]))
                    yaw = float((nose[0] - eye[0]) / scale)
                    pitch = float((mouth[1] - nose[1]) / scale)
                pose_result = pose.process(rgb)
                if getattr(pose_result, "pose_landmarks", None):
                    pl = pose_result.pose_landmarks.landmark
                    shoulders = 0.5 * (np.array([pl[11].x, pl[11].y]) + np.array([pl[12].x, pl[12].y]))
                    hips = 0.5 * (np.array([pl[23].x, pl[23].y]) + np.array([pl[24].x, pl[24].y]))
                    torso_vec = shoulders - hips
                    torso = float(np.arctan2(torso_vec[0], -torso_vec[1] + 1e-6))
                vec = np.array([yaw, pitch, roll, torso], np.float32)
                energy = 0.0 if last is None or np.any(np.isnan(vec)) or np.any(np.isnan(last)) else float(np.linalg.norm(vec - last))
                last = vec.copy()
                rows.append([yaw, pitch, roll, torso, energy])
    except Exception as exc:
        print("MediaPipe target extraction failed; optical flow fallback for", video, "|", repr(exc))
        cap.release()
        return optical_flow_targets(video)
    cap.release()
    arr = np.asarray(rows, np.float32)
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return optical_flow_targets(video)
    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx]
        if np.all(np.isnan(col)):
            col[:] = 0
        else:
            idx = np.arange(len(col))
            good = ~np.isnan(col)
            col[~good] = np.interp(idx[~good], idx[good], col[good])
            arr[:, col_idx] = col
    return np.nan_to_num(resize_seq(arr)).astype(np.float32)


def load_motion_file(path: str | Path) -> Optional[np.ndarray]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        if path.suffix == ".npy":
            arr = np.load(path)
        elif path.suffix == ".npz":
            z = np.load(path, allow_pickle=True)
            arrs = [z[k] for k in z.files if isinstance(z[k], np.ndarray) and np.issubdtype(z[k].dtype, np.number) and z[k].ndim >= 2]
            if not arrs:
                return None
            arr = max(arrs, key=lambda value: value.size)
        elif path.suffix == ".json":
            obj = json.loads(path.read_text())
            arr = np.asarray(obj.get("motion", []), np.float32)
        else:
            return None
        arr = np.asarray(arr, np.float32)
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return None
        if arr.shape[-1] < 5:
            arr = np.c_[arr, np.zeros((arr.shape[0], 5 - arr.shape[-1]), np.float32)]
        return resize_seq(arr[:, :5]).astype(np.float32)
    except Exception as exc:
        print("motion load failed", path, repr(exc))
        return None


def _cfg_payload() -> dict:
    payload = asdict(CFG)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _init_cache_worker(payload: dict) -> None:
    global CFG
    path_keys = {"project_root", "data_root", "cache_root", "output_root", "sota_root"}
    kwargs = {key: Path(value) if key in path_keys else value for key, value in payload.items()}
    CFG = Config(**kwargs)
    setup_mediapipe()


def _cache_one(item: dict, force: bool = False) -> tuple[Optional[dict], Optional[list[str]]]:
    key = hashlib.md5((str(item["dataset"]) + "_" + str(item["clip_id"])).encode()).hexdigest()
    x_path = CFG.cache_root / "X" / f"{key}.npy"
    y_path = CFG.cache_root / "Y" / f"{key}.npy"
    x_path.parent.mkdir(parents=True, exist_ok=True)
    y_path.parent.mkdir(parents=True, exist_ok=True)
    if x_path.exists() and y_path.exists() and not force:
        return {**item, "feature_path": str(x_path), "target_path": str(y_path)}, None
    try:
        wav = ensure_wav(item["audio_path"])
        x, feat_names = acoustic_features(wav)
        y = None
        motion_path = item.get("motion_path", None)
        video_path = item.get("video_path", None)
        if isinstance(motion_path, str) and Path(motion_path).exists():
            y = load_motion_file(motion_path)
        if y is None and isinstance(video_path, str) and Path(video_path).exists():
            y = mediapipe_targets(video_path)
        if y is None:
            return None, feat_names
        if np.isfinite(x).all() and np.isfinite(y).all():
            np.save(x_path, x.astype(np.float32))
            np.save(y_path, y.astype(np.float32))
            return {**item, "feature_path": str(x_path), "target_path": str(y_path)}, feat_names
    except Exception as exc:
        print("skip", item.get("clip_id"), repr(exc))
    return None, None


def build_cache(index: pd.DataFrame, limit: Optional[int] = None, force: bool = False, workers: int = 1) -> pd.DataFrame:
    df = index.copy()
    if limit:
        df = df.sample(min(limit, len(df)), random_state=SEED).reset_index(drop=True)
    rows = []
    feat_names = None
    records = df.to_dict("records")
    worker_fn = partial(_cache_one, force=force)
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_cache_worker, initargs=(_cfg_payload(),)) as pool:
            for row, names in tqdm(pool.map(worker_fn, records, chunksize=4), total=len(records), desc=f"Real feature extraction x{workers}"):
                if row is not None:
                    rows.append(row)
                if names:
                    feat_names = names
    else:
        for item in tqdm(records, total=len(records), desc="Real feature extraction"):
            row, names = _cache_one(item, force=force)
            if row is not None:
                rows.append(row)
            if names:
                feat_names = names
    out = pd.DataFrame(rows)
    out.to_csv(CFG.table_dir / "real_feature_index.csv", index=False)
    if feat_names:
        (CFG.cache_root / "feature_names.json").write_text(json.dumps(feat_names, indent=2))
    print("Cached", len(out), "clips")
    return out


class SeqDS(Dataset):
    def __init__(self, df: pd.DataFrame, xs: Optional[StandardScaler] = None, ys: Optional[StandardScaler] = None, fit: bool = False, mask: Optional[list[int]] = None):
        self.df = df.reset_index(drop=True)
        xs_in = []
        ys_in = []
        for _, row in self.df.iterrows():
            x = np.load(row.feature_path).astype(np.float32)
            y = np.load(row.target_path).astype(np.float32)
            if mask is not None:
                x = x[:, mask]
            xs_in.append(x)
            ys_in.append(y)
        self.X = np.stack(xs_in)
        self.Y_raw = np.stack(ys_in)
        n, t, d = self.X.shape
        out_dim = self.Y_raw.shape[-1]
        self.xs = StandardScaler().fit(self.X.reshape(-1, d)) if fit else xs
        self.ys = StandardScaler().fit(self.Y_raw.reshape(-1, out_dim)) if fit else ys
        if self.xs is not None:
            self.X = self.xs.transform(self.X.reshape(-1, d)).reshape(n, t, d).astype(np.float32)
        self.Y = self.Y_raw.copy()
        if self.ys is not None:
            self.Y = self.ys.transform(self.Y.reshape(-1, out_dim)).reshape(n, t, out_dim).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        if self.ys is None:
            return y
        shape = y.shape
        return self.ys.inverse_transform(y.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)


def split_by_speaker(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["speaker_id"] = df.get("speaker_id", df.clip_id).fillna(df.clip_id).astype(str)
    if df.speaker_id.nunique() < 3:
        raise RuntimeError("Need at least three speaker groups for train/val/test speaker split.")
    groups = df.speaker_id.values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_val_idx, test_idx = next(gss.split(df, groups=groups))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1875, random_state=SEED + 1)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df.speaker_id.values))
    return train_val_df.iloc[train_idx].reset_index(drop=True), train_val_df.iloc[val_idx].reset_index(drop=True), test_df


class TCNBlock(nn.Module):
    def __init__(self, dim: int, kernel: int = 5, dilation: int = 1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.c1 = nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation)
        self.c2 = nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = F.gelu(self.c1(y))
        y = F.gelu(self.c2(y)).transpose(1, 2)
        if y.shape[1] != x.shape[1]:
            y = y[:, : x.shape[1], :]
        return self.norm(x + y)


class BiGRU(nn.Module):
    def __init__(self, inp: int, out: int = 5, hidden: int = 192):
        super().__init__()
        self.rnn = nn.GRU(inp, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Linear(hidden * 2, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.head(y)


class TCN(nn.Module):
    def __init__(self, inp: int, out: int = 5, hidden: int = 192):
        super().__init__()
        self.inp = nn.Linear(inp, hidden)
        self.blocks = nn.ModuleList([TCNBlock(hidden, dilation=d) for d in [1, 2, 4, 8]])
        self.head = nn.Linear(hidden, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inp(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class TransformerModel(nn.Module):
    def __init__(self, inp: int, out: int = 5, hidden: int = 192):
        super().__init__()
        self.inp = nn.Linear(inp, hidden)
        self.pos = nn.Parameter(torch.randn(1, CFG.target_len, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden * 4, dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.head = nn.Linear(hidden, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        h = self.inp(x) + self.pos[:, :t]
        return self.head(self.encoder(h))


class AcousticPose(nn.Module):
    def __init__(self, inp: int, out: int = 5, hidden: int = 256):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.temporal = nn.ModuleList([TCNBlock(hidden, dilation=d) for d in [1, 2, 4, 8, 16]])
        self.enc = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out))
        self.reproject = nn.Sequential(nn.Linear(out, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, min(inp, 8)))

    def forward(self, x: torch.Tensor, aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.inp(x)
        for block in self.temporal:
            h = block(h)
        gate = self.gate(h)
        h = gate * self.enc(h) + (1.0 - gate) * h
        y = self.head(h)
        return (y, self.reproject(y)) if aux else y


def make_model(name: str, inp: int, out: int = 5) -> nn.Module:
    name = name.lower()
    if name == "bigru":
        return BiGRU(inp, out, hidden=CFG.hidden_size)
    if name == "tcn":
        return TCN(inp, out, hidden=CFG.hidden_size)
    if name == "transformer":
        return TransformerModel(inp, out, hidden=CFG.hidden_size)
    if name == "acousticpose":
        return AcousticPose(inp, out, hidden=CFG.hidden_size)
    raise ValueError(f"Unknown model: {name}")


def temp_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target)
    loss = loss + 0.25 * F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
    loss = loss + 0.05 * F.smooth_l1_loss(pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2], target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2])
    return loss


def train_torch(model: nn.Module, train_ds: SeqDS, val_ds: SeqDS, name: str, epochs: Optional[int] = None) -> nn.Module:
    epochs = epochs or CFG.epochs
    dev = CFG.device
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    best = float("inf")
    state = None
    bad = 0
    history = []
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(dev)
            y = y.to(dev)
            opt.zero_grad()
            pred = model(x)
            loss = temp_loss(pred, y)
            if isinstance(model, AcousticPose):
                _, reproj = model(x, aux=True)
                loss = loss + 0.03 * F.smooth_l1_loss(reproj, x[:, :, : reproj.shape[-1]])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        vals = []
        with torch.no_grad():
            for x, y in val_loader:
                vals.append(float(temp_loss(model(x.to(dev)), y.to(dev)).cpu()))
        val_loss = float(np.mean(vals))
        row = {"epoch": epoch, "train": float(np.mean(losses)), "val": val_loss}
        history.append(row)
        if epoch % 5 == 0 or epoch == 1:
            print(name, epoch, row)
        if val_loss < best:
            best = val_loss
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= CFG.patience:
            break
    if state:
        model.load_state_dict(state)
    pd.DataFrame(history).to_csv(CFG.table_dir / f"{name}_history.csv", index=False)
    torch.save(model.state_dict(), CFG.model_dir / f"{name}.pt")
    return model


def pred_torch(model: nn.Module, ds: SeqDS) -> tuple[np.ndarray, np.ndarray]:
    dev = CFG.device
    model.to(dev).eval()
    ys = []
    preds = []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=CFG.batch_size):
            preds.append(model(x.to(dev)).cpu().numpy())
            ys.append(y.numpy())
    y_scaled = np.concatenate(ys)
    p_scaled = np.concatenate(preds)
    return ds.inverse_y(y_scaled), ds.inverse_y(p_scaled)


def flat(ds: SeqDS) -> tuple[np.ndarray, np.ndarray]:
    return ds.X.reshape(-1, ds.X.shape[-1]), ds.Y.reshape(-1, ds.Y.shape[-1])


def flat_sample(ds: SeqDS, max_frames: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    x, y = flat(ds)
    if max_frames and len(x) > max_frames:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(x), size=max_frames, replace=False)
        return x[idx], y[idx]
    return x, y


def update_partial_results(rows: list[dict]) -> pd.DataFrame:
    partial = pd.DataFrame(rows).sort_values("overall_mae")
    partial.to_csv(CFG.table_dir / "main_real_results_partial.csv", index=False)
    partial.to_csv(CFG.table_dir / "main_real_results.csv", index=False)
    print("Partial results saved:", CFG.table_dir / "main_real_results_partial.csv")
    print(partial[["model", "overall_mae", "overall_rmse", "motion_event_f1"]])
    return partial


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ravel(a)
    b = np.ravel(b)
    return 0.0 if np.std(a) < 1e-8 or np.std(b) < 1e-8 else float(np.corrcoef(a, b)[0, 1])


def dtw(a: np.ndarray, b: np.ndarray, max_len: int = 200) -> float:
    a = np.ravel(a)
    b = np.ravel(b)
    if len(a) > max_len:
        idx = np.linspace(0, len(a) - 1, max_len).astype(int)
        a = a[idx]
        b = b[idx]
    table = np.full((len(a) + 1, len(b) + 1), np.inf, dtype=np.float32)
    table[0, 0] = 0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            table[i, j] = abs(a[i - 1] - b[j - 1]) + min(table[i - 1, j], table[i, j - 1], table[i - 1, j - 1])
    return float(table[-1, -1] / (len(a) + len(b)))


def event_f1(y: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for idx in range(len(y)):
        yt = y[idx, :, -1]
        yp = pred[idx, :, -1]
        target_peaks, _ = find_peaks(yt, distance=5, prominence=np.std(yt) * 0.3 if np.std(yt) > 0 else 0.01)
        pred_peaks, _ = find_peaks(yp, distance=5, prominence=np.std(yp) * 0.3 if np.std(yp) > 0 else 0.01)
        matched = 0
        used = set()
        for peak in pred_peaks:
            if len(target_peaks) == 0:
                continue
            best = int(np.argmin(np.abs(target_peaks - peak)))
            if abs(target_peaks[best] - peak) <= 3 and best not in used:
                matched += 1
                used.add(best)
        precision = matched / max(len(pred_peaks), 1)
        recall = matched / max(len(target_peaks), 1)
        vals.append(2 * precision * recall / (precision + recall + 1e-8))
    return float(np.mean(vals))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    out = {
        "overall_mae": float(np.mean(np.abs(y - pred))),
        "overall_rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "motion_event_f1": event_f1(y, pred),
    }
    for dim_idx, name in enumerate(TARGET_DIMS):
        out[f"{name}_mae"] = float(np.mean(np.abs(y[:, :, dim_idx] - pred[:, :, dim_idx])))
        out[f"{name}_corr"] = corr(y[:, :, dim_idx], pred[:, :, dim_idx])
        out[f"{name}_dtw"] = float(np.mean([dtw(y[i, :, dim_idx], pred[i, :, dim_idx]) for i in range(min(len(y), 60))]))
    return out


def paired_test(y: np.ndarray, baseline: np.ndarray, acousticpose: np.ndarray) -> dict:
    eb = np.mean(np.abs(y - baseline), axis=(1, 2))
    ea = np.mean(np.abs(y - acousticpose), axis=(1, 2))
    t_stat, p_value = stats.ttest_rel(eb, ea)
    return {
        "baseline_mae": float(eb.mean()),
        "acousticpose_mae": float(ea.mean()),
        "p": float(p_value),
        "t": float(t_stat),
        "cohen_dz": float(np.mean(eb - ea) / (np.std(eb - ea) + 1e-8)),
    }


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, n: int = 1000) -> list[float]:
    rng = np.random.default_rng(SEED)
    vals = []
    count = len(y)
    for _ in range(n):
        idx = rng.integers(0, count, count)
        vals.append(np.mean(np.abs(y[idx] - pred[idx])))
    return np.percentile(vals, [2.5, 50, 97.5]).tolist()


def run_training(
    feature_index: pd.DataFrame,
    models: list[str],
    run_ablations: bool,
    tree_jobs: int = 1,
    tree_estimators: int = 60,
    rf_estimators: int = 50,
    max_sklearn_frames: Optional[int] = 300_000,
    train_clip_limit: Optional[int] = None,
    val_clip_limit: Optional[int] = None,
    resume_partial: bool = True,
    ablation_epochs: Optional[int] = None,
) -> None:
    train_df, val_df, test_df = split_by_speaker(feature_index)
    split_df = pd.concat([train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")])
    split_df.to_csv(CFG.table_dir / "speaker_split.csv", index=False)
    print("Split sizes:", len(train_df), len(val_df), len(test_df))
    print(split_df.groupby(["split", "dataset"]).size())
    full_train_n = len(train_df)
    full_val_n = len(val_df)
    if train_clip_limit and len(train_df) > train_clip_limit:
        train_df = train_df.sample(train_clip_limit, random_state=SEED).reset_index(drop=True)
        print("Training clip cap applied:", len(train_df), "of", full_train_n)
    if val_clip_limit and len(val_df) > val_clip_limit:
        val_df = val_df.sample(val_clip_limit, random_state=SEED).reset_index(drop=True)
        print("Validation clip cap applied:", len(val_df), "of", full_val_n)

    train_ds = SeqDS(train_df, fit=True)
    val_ds = SeqDS(val_df, xs=train_ds.xs, ys=train_ds.ys)
    test_ds = SeqDS(test_df, xs=train_ds.xs, ys=train_ds.ys)
    inp = train_ds.X.shape[-1]
    out_dim = len(TARGET_DIMS)
    print("Shapes:", train_ds.X.shape, val_ds.X.shape, test_ds.X.shape)
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    partial_path = CFG.table_dir / "main_real_results_partial.csv"
    completed = set()
    if resume_partial and partial_path.exists():
        partial = pd.read_csv(partial_path)
        rows = partial.to_dict("records")
        completed = set(partial.model.astype(str))
        print("Resuming partial model results:", sorted(completed))

    baseline_specs = [
        ("ridge_frame", Ridge(alpha=1.0)),
        ("extra_trees_frame", ExtraTreesRegressor(n_estimators=tree_estimators, n_jobs=tree_jobs, random_state=SEED)),
        ("random_forest_frame", RandomForestRegressor(n_estimators=rf_estimators, max_depth=18, n_jobs=tree_jobs, random_state=SEED)),
    ]
    for name, model in baseline_specs:
        if name not in models:
            continue
        if name in completed:
            print("Skipping completed", name)
            continue
        print("Training", name)
        x_train, y_train = flat_sample(train_ds, max_sklearn_frames)
        print("Sklearn fit frames:", len(x_train), "tree_jobs:", tree_jobs)
        model.fit(x_train, y_train)
        pred_scaled = model.predict(flat(test_ds)[0]).reshape(test_ds.Y.shape).astype(np.float32)
        y = test_ds.Y_raw
        pred = test_ds.inverse_y(pred_scaled)
        preds[name] = (y, pred)
        rows.append({"model": name, "train_clips_used": len(train_df), "val_clips_used": len(val_df), "test_clips_used": len(test_df), **metrics(y, pred)})
        update_partial_results(rows)
        del model, x_train, y_train, pred_scaled, y, pred
        gc.collect()

    for name in ["bigru", "tcn", "transformer", "acousticpose"]:
        if name not in models:
            continue
        if name in completed:
            print("Skipping completed", name)
            continue
        print("Training", name)
        model = train_torch(make_model(name, inp, out_dim), train_ds, val_ds, name)
        y, pred = pred_torch(model, test_ds)
        preds[name] = (y, pred)
        rows.append({"model": name, "train_clips_used": len(train_df), "val_clips_used": len(val_df), "test_clips_used": len(test_df), **metrics(y, pred)})
        update_partial_results(rows)
        del model
        gc.collect()

    main = pd.DataFrame(rows).sort_values("overall_mae")
    main.to_csv(CFG.table_dir / "main_real_results.csv", index=False)
    print(main)

    ablation_df = pd.DataFrame()
    if run_ablations and "acousticpose" in models:
        feature_names = json.loads((CFG.cache_root / "feature_names.json").read_text()) if (CFG.cache_root / "feature_names.json").exists() else []
        ablation_path = CFG.table_dir / "ablation_real_results.csv"
        ablation_rows = []
        completed_ablations = set()
        if resume_partial and ablation_path.exists():
            partial_ablation = pd.read_csv(ablation_path)
            ablation_rows = partial_ablation.to_dict("records")
            completed_ablations = set(partial_ablation.ablation.astype(str))
            print("Resuming partial ablation results:", sorted(completed_ablations))

        def excl(keys: list[str]) -> list[int]:
            return [i for i, name in enumerate(feature_names) if not any(key in name.lower() for key in keys)]

        def incl(keys: list[str]) -> list[int]:
            return [i for i, name in enumerate(feature_names) if any(key in name.lower() for key in keys)]

        def incl_excl(include_keys: list[str], exclude_keys: list[str]) -> list[int]:
            return [i for i, name in enumerate(feature_names) if any(key in name.lower() for key in include_keys) and not any(key in name.lower() for key in exclude_keys)]

        ablations = {
            "full_acousticpose": None,
            "no_high_frequency": excl(["hf_ratio", "mid_high", "hf_drift"]),
            "no_rolloff_centroid": excl(["centroid", "rolloff"]),
            "no_spectral_shape": excl(["bandwidth", "flatness", "zcr"]),
            "no_drr": excl(["drr"]),
            "no_off_axis": excl(["off_axis"]),
            "no_plosive": excl(["plosive"]),
            "no_silence_texture": excl(["silence"]),
            "no_temporal_drift": excl(["drift"]),
            "no_energy_dynamics": excl(["rms", "flux", "level_drift"]),
            "geometry_only": incl(["hf", "centroid", "rolloff", "drr", "off_axis", "level_drift"]),
            "spectral_only": incl(["hf", "mid_high", "centroid", "rolloff", "bandwidth", "flatness"]),
            "proxy_cues_only": incl(["drr", "off_axis", "plosive", "silence"]),
            "speech_energy_only": incl(["rms", "flux", "zcr"]),
            "no_plosive_no_temporal_drift": excl(["plosive", "drift"]),
            "no_plosive_no_rolloff_centroid": excl(["plosive", "centroid", "rolloff"]),
            "no_plosive_no_energy_dynamics": excl(["plosive", "rms", "flux", "level_drift"]),
            "no_temporal_drift_no_energy_dynamics": excl(["drift", "rms", "flux"]),
            "no_temporal_drift_no_rolloff_centroid": excl(["drift", "centroid", "rolloff"]),
            "no_drr_no_plosive": excl(["drr", "plosive"]),
            "no_drr_no_rolloff_centroid": excl(["drr", "centroid", "rolloff"]),
            "no_high_frequency_no_temporal_drift": excl(["hf_ratio", "mid_high", "hf_drift", "drift"]),
            "geometry_no_drift": incl_excl(["hf", "centroid", "rolloff", "drr", "off_axis", "level_drift"], ["drift"]),
            "geometry_plus_speech_energy": incl(["hf", "centroid", "rolloff", "drr", "off_axis", "rms", "flux", "zcr"]),
            "geometry_plus_silence": incl(["hf", "centroid", "rolloff", "drr", "off_axis", "silence"]),
            "compact_no_plosive_drift_rolloff": excl(["plosive", "drift", "centroid", "rolloff"]),
        }
        for feature_idx, feature_name in enumerate(feature_names):
            safe_name = "".join(ch if ch.isalnum() else "_" for ch in feature_name.lower()).strip("_")
            ablations[f"drop_{safe_name}"] = [i for i in range(len(feature_names)) if i != feature_idx]
        run_epochs = ablation_epochs or max(3, CFG.epochs // 2)
        for name, mask in ablations.items():
            if name in completed_ablations:
                print("Skipping completed ablation", name)
                continue
            if mask is not None and len(mask) < 2:
                continue
            print("Ablation", name, "features", "all" if mask is None else len(mask))
            tr = SeqDS(train_df, fit=True, mask=mask)
            va = SeqDS(val_df, xs=tr.xs, ys=tr.ys, mask=mask)
            te = SeqDS(test_df, xs=tr.xs, ys=tr.ys, mask=mask)
            model = train_torch(make_model("acousticpose", tr.X.shape[-1], out_dim), tr, va, "ablation_" + name, epochs=run_epochs)
            y, pred = pred_torch(model, te)
            ablation_rows.append({"ablation": name, **metrics(y, pred)})
            ablation_df = pd.DataFrame(ablation_rows).sort_values("overall_mae")
            ablation_df.to_csv(ablation_path, index=False)
            print("Partial ablations saved:", ablation_path)
            print(ablation_df[["ablation", "overall_mae", "overall_rmse", "motion_event_f1"]])
            del tr, va, te, model, y, pred
            gc.collect()
        ablation_df = pd.DataFrame(ablation_rows).sort_values("overall_mae")
        ablation_df.to_csv(ablation_path, index=False)
        print(ablation_df)

    if "acousticpose" in preds and len(main[main.model != "acousticpose"]) > 0:
        ap_y, ap_pred = preds["acousticpose"]
        best = main[main.model != "acousticpose"].sort_values("overall_mae").iloc[0].model
        if best not in preds:
            print("Skipping significance report because best baseline predictions are not in memory after partial resume:", best)
            write_figures(preds)
            return
        _, best_pred = preds[best]
        sig = {
            **paired_test(ap_y, best_pred, ap_pred),
            "best_baseline": best,
            "acousticpose_ci95": json.dumps(bootstrap_ci(ap_y, ap_pred)),
            "baseline_ci95": json.dumps(bootstrap_ci(ap_y, best_pred)),
        }
        pd.DataFrame([sig]).to_csv(CFG.table_dir / "significance_report.csv", index=False)
        rel = (sig["baseline_mae"] - sig["acousticpose_mae"]) / (sig["baseline_mae"] + 1e-8)
        if len(ablation_df) and "full_acousticpose" in set(ablation_df.ablation):
            full = ablation_df[ablation_df.ablation == "full_acousticpose"].iloc[0].overall_mae
            reduced = ablation_df[ablation_df.ablation != "full_acousticpose"].overall_mae.min()
            ablation_gain = (reduced - full) / (reduced + 1e-8)
        else:
            ablation_gain = 0.0
        gate = {
            "verdict_AAAI_reviewer_proof": bool(
                len(feature_index) >= CFG.min_clips
                and feature_index.dataset.nunique() >= CFG.min_datasets
                and rel >= CFG.min_rel_impr
                and ablation_gain >= CFG.min_ablation_gain
                and sig["p"] < 0.05
                and sig["cohen_dz"] > 0.5
            ),
            "n_clips": len(feature_index),
            "n_datasets": feature_index.dataset.nunique(),
            "datasets": json.dumps(feature_index.dataset.value_counts().to_dict()),
            "best_baseline": best,
            "relative_improvement": float(rel),
            "ablation_gain": float(ablation_gain),
            "p_value": sig["p"],
            "cohen_dz": sig["cohen_dz"],
        }
        pd.DataFrame([gate]).to_csv(CFG.table_dir / "reviewer_proof_gate.csv", index=False)
        print(pd.DataFrame([gate]))
        print("REVIEWER-PROOF GATE PASSED" if gate["verdict_AAAI_reviewer_proof"] else "Gate failed. Do not claim paper-ready results yet.")

    write_figures(preds)


def write_figures(preds: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    main_path = CFG.table_dir / "main_real_results.csv"
    if main_path.exists():
        main = pd.read_csv(main_path).sort_values("overall_mae")
        plt.figure(figsize=(12, 5))
        plt.bar(main.model, main.overall_mae)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Overall MAE lower is better")
        plt.title("Real-data motion reconstruction")
        plt.tight_layout()
        plt.savefig(CFG.fig_dir / "main_real_mae.png", dpi=240)
        plt.close()
    gate_path = CFG.table_dir / "reviewer_proof_gate.csv"
    if "acousticpose" in preds and gate_path.exists():
        gate = pd.read_csv(gate_path).iloc[0]
        best = gate.best_baseline
        if best in preds:
            y, pred = preds["acousticpose"]
            _, baseline = preds[best]
            plt.figure(figsize=(13, 4))
            plt.plot(y[0, :, 0], label="ground truth")
            plt.plot(baseline[0, :, 0], label=f"best baseline: {best}", alpha=0.75)
            plt.plot(pred[0, :, 0], label="AcousticPose", alpha=0.9)
            plt.title("Example real trajectory reconstruction")
            plt.legend()
            plt.tight_layout()
            plt.savefig(CFG.fig_dir / "example_real_trajectory.png", dpi=240)
            plt.close()


def eval_external(test_df: pd.DataFrame) -> None:
    rows = []
    for model_name in [
        "TaoAvatar_driver_projection",
        "CyberHost",
        "Audio2Photoreal",
        "TalkSHOW",
        "DiffTED",
        "BEAT_CaMN",
        "GestureDiffuCLIP",
        "EMAGE",
        "DEEPTalk_EchoMimic",
    ]:
        folder = CFG.sota_root / model_name
        if not folder.exists():
            continue
        ys = []
        preds = []
        for _, row in test_df.iterrows():
            pred_path = folder / f"{row.clip_id}.npy"
            if pred_path.exists():
                ys.append(np.load(row.target_path))
                preds.append(resize_seq(np.load(pred_path)))
        if len(ys) >= 10:
            y = np.stack(ys)
            pred = np.stack(preds)
            rows.append({"model": model_name, "n_clips": len(ys), **metrics(y, pred)})
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(CFG.table_dir / "external_sota_results.csv", index=False)


def save_config(args: argparse.Namespace) -> None:
    cfg_dict = asdict(CFG)
    for key, value in cfg_dict.items():
        if isinstance(value, Path):
            cfg_dict[key] = str(value)
    cfg_dict["cli_args"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (CFG.output_root / "run_config.json").write_text(json.dumps(cfg_dict, indent=2))


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AcousticPose locally on real audio-video/motion data.")
    parser.add_argument("--project-root", type=Path, default=Path("acousticpose_runs/local"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sota-root", type=Path, default=None)
    parser.add_argument("--download-ravdess", choices=["none", "subset", "full"], default="none")
    parser.add_argument("--ravdess-actors", default="1-8", help="Actor list/range for RAVDESS subset, e.g. 1-8 or 1,2,3")
    parser.add_argument("--download-cremad", action="store_true")
    parser.add_argument("--download-meld", action="store_true")
    parser.add_argument("--beat-root", type=Path)
    parser.add_argument("--beat2-root", type=Path)
    parser.add_argument("--talkshow-root", type=Path)
    parser.add_argument("--audio2photoreal-root", type=Path)
    parser.add_argument("--av-root", type=Path, help="Optional folder of ordinary local videos for development experiments.")
    parser.add_argument("--av-dataset-name", default="LOCAL_AV")
    parser.add_argument("--target-backend", choices=["auto", "mediapipe", "optical_flow"], default="auto")
    parser.add_argument("--feature-limit", type=int)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Parallel feature/target extraction workers. Use 1 for sequential extraction.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size for neural sequence models.")
    parser.add_argument("--models", default="ridge_frame,extra_trees_frame,random_forest_frame,bigru,tcn,transformer,acousticpose")
    parser.add_argument("--tree-jobs", type=int, default=1, help="CPU workers for sklearn tree baselines.")
    parser.add_argument("--tree-estimators", type=int, default=60, help="ExtraTrees estimator count.")
    parser.add_argument("--rf-estimators", type=int, default=50, help="RandomForest estimator count.")
    parser.add_argument("--max-sklearn-frames", type=int, default=300000, help="Maximum sampled frame rows for sklearn baseline fitting.")
    parser.add_argument("--train-clip-limit", type=int, default=None, help="Optional sampled training clip cap for low-resource training.")
    parser.add_argument("--val-clip-limit", type=int, default=None, help="Optional sampled validation clip cap for low-resource training.")
    parser.add_argument("--no-resume-partial", action="store_true", help="Do not skip models already present in the partial results table.")
    parser.add_argument("--ablation-epochs", type=int, default=None, help="Epochs per ablation model. Defaults to a short staged run based on --epochs.")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-clips", type=int, default=500)
    parser.add_argument("--min-datasets", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Fast local smoke-test settings. This disables strict paper claims.")
    return parser


def parse_actors(spec: str) -> list[int]:
    actors = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            actors.extend(range(int(a), int(b) + 1))
        elif part:
            actors.append(int(part))
    return sorted(set(actors))


def main() -> None:
    global CFG
    args = make_arg_parser().parse_args()
    seed_everything()
    project_root = args.project_root.resolve()
    CFG = Config(
        project_root=project_root,
        data_root=(args.data_root or project_root / "data").resolve(),
        cache_root=(args.cache_root or project_root / "cache").resolve(),
        output_root=(args.output_root or project_root / "outputs").resolve(),
        sota_root=(args.sota_root or project_root / "sota_outputs").resolve(),
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        strict=args.strict,
        min_clips=args.min_clips,
        min_datasets=args.min_datasets,
        target_backend=args.target_backend,
        hidden_size=args.hidden_size,
    )
    if args.quick:
        CFG.epochs = min(CFG.epochs, 2)
        CFG.patience = min(CFG.patience, 2)
        CFG.batch_size = min(CFG.batch_size, 8)
        CFG.strict = False
        args.feature_limit = args.feature_limit or 12
        args.models = "ridge_frame,acousticpose"
        args.skip_ablations = True
    CFG.mkdirs()
    save_config(args)
    require_ffmpeg()
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in asdict(CFG).items()}, indent=2))
    print("Torch", torch.__version__, "CUDA", torch.cuda.is_available(), "device", CFG.device)

    if args.download_ravdess == "full":
        download_ravdess(list(range(1, 25)))
    elif args.download_ravdess == "subset":
        download_ravdess(parse_actors(args.ravdess_actors))
    if args.download_cremad:
        clone_cremad()
    if args.download_meld:
        download_meld_raw()

    mp_status = setup_mediapipe()
    print("Motion target backend:", mp_status)

    master_index = build_master_index(args)
    master_index.to_csv(CFG.table_dir / "master_index.csv", index=False)
    print("Indexed real clips:", len(master_index))
    if len(master_index):
        print(master_index.dataset.value_counts())
    if CFG.strict and len(master_index) < 50:
        raise RuntimeError("Real-data gate failed: download/mount real datasets. No synthetic results are allowed.")

    feature_index = build_cache(master_index, limit=args.feature_limit, force=args.force_cache, workers=args.workers)
    if len(feature_index):
        print(feature_index.dataset.value_counts())
    if CFG.strict and len(feature_index) < 50:
        raise RuntimeError("Not enough real cached clips. No synthetic/preliminary claims allowed.")
    if len(feature_index) < 6:
        raise RuntimeError("Need at least 6 cached clips for a local smoke training run.")

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    run_training(
        feature_index,
        models=models,
        run_ablations=not args.skip_ablations,
        tree_jobs=max(1, args.tree_jobs),
        tree_estimators=args.tree_estimators,
        rf_estimators=args.rf_estimators,
        max_sklearn_frames=args.max_sklearn_frames,
        train_clip_limit=args.train_clip_limit,
        val_clip_limit=args.val_clip_limit,
        resume_partial=not args.no_resume_partial,
        ablation_epochs=args.ablation_epochs,
    )
    print("Outputs written to:", CFG.output_root)


if __name__ == "__main__":
    main()
