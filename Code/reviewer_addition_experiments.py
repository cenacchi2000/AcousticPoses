#!/usr/bin/env python3
"""Additional reviewer-critical AcousticPose experiments.

This script is intentionally modular and CPU-light. It uses existing cached
features/checkpoints where possible and writes partial CSVs after each block.
It does not fabricate unavailable OpenFace/MediaPipe/mocap/SOTA results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from scipy import signal, stats
from sklearn.linear_model import Ridge

import acousticpose_local as ap
from rigorous_evaluation_extensions import facebox_proxy, per_clip_event_f1, per_clip_mae


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "work/full_public_stage1"
RESULTS = ROOT / "outputs/full_public_stage1_results"
SEED = 42


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
    ap.seed_everything(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ap.SeqDS, ap.SeqDS, ap.SeqDS]:
    feature_index = pd.read_csv(ap.CFG.table_dir / "real_feature_index.csv")
    train_df, val_df, test_df = ap.split_by_speaker(feature_index)
    train_ds = ap.SeqDS(train_df, fit=True)
    val_ds = ap.SeqDS(val_df, xs=train_ds.xs, ys=train_ds.ys)
    test_ds = ap.SeqDS(test_df, xs=train_ds.xs, ys=train_ds.ys)
    return train_df, val_df, test_df, train_ds, val_ds, test_ds


def load_model(name: str, state_file: str, inp: int) -> torch.nn.Module:
    model = ap.make_model(name, inp, len(ap.TARGET_DIMS))
    model.load_state_dict(torch.load(ap.CFG.model_dir / state_file, map_location="cpu"))
    model.eval()
    return model


def predict_on_x(model: torch.nn.Module, ds: ap.SeqDS, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    old = ds.X
    ds.X = x.astype(np.float32)
    try:
        return ap.pred_torch(model, ds)
    finally:
        ds.X = old


def fit_ridge(train_ds: ap.SeqDS, max_frames: int = 150_000) -> Ridge:
    x, y = ap.flat_sample(train_ds, max_frames)
    ridge = Ridge(alpha=1.0)
    ridge.fit(x, y)
    return ridge


def ridge_predict(ridge: Ridge, ds: ap.SeqDS, x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x_eval = ds.X if x is None else x
    pred_scaled = ridge.predict(x_eval.reshape(-1, x_eval.shape[-1])).reshape(ds.Y.shape).astype(np.float32)
    return ds.Y_raw, ds.inverse_y(pred_scaled)


def same_group_wrong_x(test_df: pd.DataFrame, x: np.ndarray, column: str, rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    values = test_df[column].fillna("__missing__").astype(str).to_numpy()
    dataset = test_df["dataset"].astype(str).to_numpy()
    for value in sorted(set(values)):
        ids = np.flatnonzero(values == value)
        if len(ids) > 1:
            out[ids] = x[rng.permutation(ids)]
    # Fall back to within-dataset permutations where a group has only one clip.
    unchanged = np.flatnonzero(np.all(out == x, axis=(1, 2)))
    for ds_name in sorted(set(dataset[unchanged])):
        ids = unchanged[dataset[unchanged] == ds_name]
        if len(ids) > 1:
            out[ids] = x[rng.permutation(ids)]
    return out


def run_negative_controls(test_df: pd.DataFrame, train_ds: ap.SeqDS, test_ds: ap.SeqDS) -> pd.DataFrame:
    out_path = RESULTS / "negative_controls.csv"
    if out_path.exists():
        print("negative controls already exist:", out_path)
        return pd.read_csv(out_path)
    rng = np.random.default_rng(SEED)
    feature_names = json.loads((ap.CFG.cache_root / "feature_names.json").read_text())
    rms_idx = feature_names.index("rms")
    x = test_ds.X.copy()
    conditions: dict[str, np.ndarray] = {"original_audio": x}
    conditions["random_clip_audio"] = x[rng.permutation(len(x))]
    conditions["same_speaker_wrong_utterance"] = same_group_wrong_x(test_df, x, "speaker_id", rng)
    if "label_emotion" in test_df.columns:
        conditions["same_emotion_wrong_speaker"] = same_group_wrong_x(test_df, x, "label_emotion", rng)
    conditions["time_reversed_audio_features"] = x[:, ::-1, :]
    rms_only = np.zeros_like(x)
    rms_only[:, :, rms_idx] = x[:, :, rms_idx]
    conditions["rms_only_features"] = rms_only

    models = {
        "ridge_frame": fit_ridge(train_ds),
        "AcousticPose-FullTCN": load_model("acousticpose", "fulltrain_acousticpose.pt", test_ds.X.shape[-1]),
        "AcousticPose-Transformer++": load_model("transformer", "fulltrain_transformer.pt", test_ds.X.shape[-1]),
    }
    rows: list[dict] = []
    for model_name, model in models.items():
        for condition, cond_x in conditions.items():
            print("negative", model_name, condition, flush=True)
            if model_name == "ridge_frame":
                y, pred = ridge_predict(model, test_ds, cond_x)
            else:
                y, pred = predict_on_x(model, test_ds, cond_x)
            rows.append({"model": model_name, "condition": condition, "n_clips": len(test_ds), **ap.metrics(y, pred)})
            pd.DataFrame(rows).to_csv(out_path, index=False)
    return pd.DataFrame(rows)


def feature_from_signal(y: np.ndarray, sr: int) -> np.ndarray:
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / (np.max(np.abs(y)) + 1e-8)
    hop = max(1, int(sr / ap.CFG.fps))
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
    return np.nan_to_num(ap.resize_seq(feats)).astype(np.float32)


def corrupt_audio(y: np.ndarray, sr: int, condition: str, rng: np.random.Generator) -> np.ndarray:
    if condition == "clean":
        return y
    if condition.startswith("noise_"):
        snr = float(condition.split("_")[1].replace("db", ""))
        power = np.mean(y**2) + 1e-8
        noise_power = power / (10 ** (snr / 10.0))
        return y + rng.normal(0, math.sqrt(noise_power), size=y.shape).astype(np.float32)
    if condition == "telephone_bandpass":
        b, a = signal.butter(4, [300 / (sr / 2), 3400 / (sr / 2)], btype="band")
        return signal.lfilter(b, a, y).astype(np.float32)
    if condition == "reverb_rt60_proxy":
        decay = np.exp(-np.linspace(0, 4.0, int(sr * 0.6), dtype=np.float32))
        impulse = np.r_[1.0, 0.35 * decay]
        return signal.fftconvolve(y, impulse, mode="full")[: len(y)].astype(np.float32)
    if condition == "packet_loss_10":
        out = y.copy()
        frame = int(0.05 * sr)
        for start in range(0, len(out), frame):
            if rng.random() < 0.10:
                out[start : start + frame] = 0
        return out
    if condition == "gain_shift_6db":
        return (y * 2.0).astype(np.float32)
    raise ValueError(condition)


def run_raw_audio_robustness(test_df: pd.DataFrame, train_ds: ap.SeqDS, test_ds: ap.SeqDS, sample_per_dataset: int = 120) -> pd.DataFrame:
    out_path = RESULTS / "raw_audio_robustness_subset.csv"
    if out_path.exists():
        print("raw-audio robustness already exists:", out_path)
        return pd.read_csv(out_path)
    rng = np.random.default_rng(SEED)
    sample = (
        test_df.groupby("dataset", group_keys=False)
        .apply(lambda df: df.sample(min(sample_per_dataset, len(df)), random_state=SEED))
        .reset_index()
        .rename(columns={"index": "test_idx"})
    )
    conditions = ["clean", "noise_20db", "noise_10db", "telephone_bandpass", "reverb_rt60_proxy", "packet_loss_10", "gain_shift_6db"]
    models = {
        "ridge_frame": fit_ridge(train_ds),
        "AcousticPose-Transformer++": load_model("transformer", "fulltrain_transformer.pt", test_ds.X.shape[-1]),
    }
    rows: list[dict] = []
    for condition in conditions:
        xs = []
        ys = []
        print("raw-audio condition", condition, flush=True)
        for _, row in sample.iterrows():
            wav = ap.ensure_wav(row.audio_path)
            y_audio, sr = librosa.load(str(wav), sr=ap.CFG.sr, mono=True, duration=ap.CFG.max_clip_sec)
            x = feature_from_signal(corrupt_audio(y_audio.astype(np.float32), sr, condition, rng), sr)
            xs.append(x)
            ys.append(np.load(row.target_path).astype(np.float32))
        x_raw = np.stack(xs)
        y_raw = np.stack(ys)
        n, t, d = x_raw.shape
        x_scaled = train_ds.xs.transform(x_raw.reshape(-1, d)).reshape(n, t, d).astype(np.float32)
        mini = ap.SeqDS(sample, xs=train_ds.xs, ys=train_ds.ys)
        mini.X = x_scaled
        mini.Y_raw = y_raw
        mini.Y = train_ds.ys.transform(y_raw.reshape(-1, y_raw.shape[-1])).reshape(y_raw.shape).astype(np.float32)
        for model_name, model in models.items():
            if model_name == "ridge_frame":
                y, pred = ridge_predict(model, mini, x_scaled)
            else:
                y, pred = predict_on_x(model, mini, x_scaled)
            clean_ref = None
            rows.append({"condition": condition, "model": model_name, "n_clips": len(mini), **ap.metrics(y, pred)})
            pd.DataFrame(rows).to_csv(out_path, index=False)
    out = pd.DataFrame(rows)
    clean = out[out.condition == "clean"][["model", "overall_mae"]].rename(columns={"overall_mae": "clean_mae"})
    out = out.merge(clean, on="model", how="left")
    out["relative_mae_degradation"] = (out.overall_mae - out.clean_mae) / (out.clean_mae + 1e-8)
    out.to_csv(out_path, index=False)
    return out


def mfcc_sequence(audio_path: str | Path) -> np.ndarray:
    wav = ap.ensure_wav(audio_path)
    y, sr = librosa.load(str(wav), sr=ap.CFG.sr, mono=True, duration=ap.CFG.max_clip_sec)
    hop = max(1, int(sr / ap.CFG.fps))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop, n_fft=1024)
    width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1)
    if width < 3:
        delta = np.zeros_like(mfcc)
        delta2 = np.zeros_like(mfcc)
    else:
        delta = librosa.feature.delta(mfcc, width=width)
        delta2 = librosa.feature.delta(mfcc, order=2, width=width)
    feats = np.concatenate([mfcc, delta, delta2], axis=0).T
    return np.nan_to_num(ap.resize_seq(feats)).astype(np.float32)


def run_mfcc_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame, train_ds: ap.SeqDS, test_ds: ap.SeqDS, train_limit: int = 3000) -> pd.DataFrame:
    out_path = RESULTS / "classic_audio_representation_baselines.csv"
    if out_path.exists():
        print("MFCC baselines already exist:", out_path)
        return pd.read_csv(out_path)
    rng = np.random.default_rng(SEED)
    tr = train_df.sample(min(train_limit, len(train_df)), random_state=SEED).reset_index(drop=True)
    frames_x = []
    frames_y = []
    for _, row in tr.iterrows():
        frames_x.append(mfcc_sequence(row.audio_path))
        frames_y.append(np.load(row.target_path).astype(np.float32))
    x_train = np.stack(frames_x)
    y_train = np.stack(frames_y)
    x_test = np.stack([mfcc_sequence(row.audio_path) for _, row in test_df.iterrows()])
    y_test = test_ds.Y_raw
    sx = ap.StandardScaler().fit(x_train.reshape(-1, x_train.shape[-1]))
    sy = ap.StandardScaler().fit(y_train.reshape(-1, y_train.shape[-1]))
    xtr = sx.transform(x_train.reshape(-1, x_train.shape[-1]))
    ytr = sy.transform(y_train.reshape(-1, y_train.shape[-1]))
    if len(xtr) > 150_000:
        ids = rng.choice(len(xtr), 150_000, replace=False)
        xtr = xtr[ids]
        ytr = ytr[ids]
    ridge = Ridge(alpha=1.0).fit(xtr, ytr)
    xte = sx.transform(x_test.reshape(-1, x_test.shape[-1]))
    pred = sy.inverse_transform(ridge.predict(xte)).reshape(y_test.shape).astype(np.float32)
    rows = [
        {"representation": "MFCC+deltas ridge-frame", "feature_dim": x_train.shape[-1], "n_train_clips": len(tr), "n_test_clips": len(test_df), **ap.metrics(y_test, pred)}
    ]
    # Include exact cached descriptor baselines for context.
    rows.append(
        {
            "representation": "AcousticPose descriptors ridge-frame",
            "feature_dim": train_ds.X.shape[-1],
            "n_train_clips": len(train_ds),
            "n_test_clips": len(test_ds),
            **ap.metrics(*ridge_predict(fit_ridge(train_ds), test_ds)),
        }
    )
    transformer = load_model("transformer", "fulltrain_transformer.pt", test_ds.X.shape[-1])
    rows.append(
        {
            "representation": "AcousticPose descriptors Transformer++",
            "feature_dim": train_ds.X.shape[-1],
            "n_train_clips": len(train_ds),
            "n_test_clips": len(test_ds),
            **ap.metrics(*predict_on_x(transformer, test_ds, test_ds.X)),
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def run_proxy_confidence(test_df: pd.DataFrame, train_ds: ap.SeqDS, test_ds: ap.SeqDS, sample_per_dataset: int = 36) -> pd.DataFrame:
    out_path = RESULTS / "proxy_confidence_bins.csv"
    detail_path = RESULTS / "proxy_confidence_detail.csv"
    if out_path.exists():
        print("proxy-confidence bins already exist:", out_path)
        return pd.read_csv(out_path)
    sample = (
        test_df.groupby("dataset", group_keys=False)
        .apply(lambda df: df.sample(min(sample_per_dataset, len(df)), random_state=SEED))
        .reset_index()
        .rename(columns={"index": "test_idx"})
    )
    model = load_model("transformer", "fulltrain_transformer.pt", test_ds.X.shape[-1])
    ridge = fit_ridge(train_ds)
    y_all, tr_pred = predict_on_x(model, test_ds, test_ds.X)
    _, ridge_pred = ridge_predict(ridge, test_ds)
    rows = []
    for _, row in sample.iterrows():
        idx = int(row.test_idx)
        cached = np.load(row.target_path).astype(np.float32)
        facebox, miss = facebox_proxy(row.video_path, target_len=cached.shape[0])
        a = cached[:, -1]
        b = facebox[:, -1]
        rho = float(stats.spearmanr(a, b).correlation) if np.std(a) >= 1e-8 and np.std(b) >= 1e-8 else 0.0
        confidence = max(0.0, (1.0 - float(miss)) * max(0.0, rho))
        rows.append(
            {
                "test_idx": idx,
                "dataset": row.dataset,
                "clip_id": row.clip_id,
                "motion_energy_spearman": rho,
                "facebox_miss_rate": float(miss),
                "proxy_confidence": confidence,
                "ridge_mae": float(np.mean(np.abs(y_all[idx] - ridge_pred[idx]))),
                "transformer_mae": float(np.mean(np.abs(y_all[idx] - tr_pred[idx]))),
                "ridge_energy_mae": float(np.mean(np.abs(y_all[idx, :, -1] - ridge_pred[idx, :, -1]))),
                "transformer_energy_mae": float(np.mean(np.abs(y_all[idx, :, -1] - tr_pred[idx, :, -1]))),
                "transformer_event_f1": float(per_clip_event_f1(y_all[idx : idx + 1], tr_pred[idx : idx + 1])[0]),
            }
        )
        pd.DataFrame(rows).to_csv(detail_path, index=False)
    detail = pd.DataFrame(rows)
    try:
        detail["confidence_bin"] = pd.qcut(detail.proxy_confidence.rank(method="first"), 3, labels=["low", "medium", "high"])
    except ValueError:
        detail["confidence_bin"] = "all"
    grouped = (
        detail.groupby("confidence_bin", observed=True)
        .agg(
            clips=("clip_id", "count"),
            confidence=("proxy_confidence", "mean"),
            miss_rate=("facebox_miss_rate", "mean"),
            motion_energy_spearman=("motion_energy_spearman", "mean"),
            ridge_mae=("ridge_mae", "mean"),
            transformer_mae=("transformer_mae", "mean"),
            ridge_energy_mae=("ridge_energy_mae", "mean"),
            transformer_energy_mae=("transformer_energy_mae", "mean"),
            transformer_event_f1=("transformer_event_f1", "mean"),
        )
        .reset_index()
    )
    detail.to_csv(detail_path, index=False)
    grouped.to_csv(out_path, index=False)
    return grouped


def write_summary() -> None:
    summary_path = RESULTS / "reviewer_additions_summary.md"
    parts = ["# Reviewer Addition Experiments\n"]
    for filename in [
        "negative_controls.csv",
        "raw_audio_robustness_subset.csv",
        "classic_audio_representation_baselines.csv",
        "proxy_confidence_bins.csv",
    ]:
        path = RESULTS / filename
        if path.exists():
            df = pd.read_csv(path)
            parts.append(f"\n## {filename}\n\n")
            cols = [c for c in ["model", "condition", "representation", "confidence_bin", "n_clips", "clips", "overall_mae", "motion_event_f1", "motion_energy_mae", "relative_mae_degradation"] if c in df.columns]
            parts.append("```text")
            parts.append(df[cols].to_string(index=False))
            parts.append("```")
            parts.append("\n")
    summary_path.write_text("\n".join(parts))


def main() -> None:
    configure()
    train_df, val_df, test_df, train_ds, val_ds, test_ds = load_splits()
    print("split", len(train_df), len(val_df), len(test_df), flush=True)
    run_negative_controls(test_df, train_ds, test_ds)
    run_proxy_confidence(test_df, train_ds, test_ds)
    run_raw_audio_robustness(test_df, train_ds, test_ds)
    run_mfcc_baselines(train_df, test_df, train_ds, test_ds)
    write_summary()


if __name__ == "__main__":
    main()
