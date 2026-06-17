#!/usr/bin/env python3
"""Reviewer-critical extensions for AcousticPose.

This script adds low-resource evidence that is missing from the main cache-only
tables: raw-video proxy consistency, deterministic qualitative cases, mechanism
stress tests, repeated seed checks for shallow baselines, and stronger simple
baselines. It writes every block as a CSV/PNG so interrupted runs keep partial
results.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

import acousticpose_local as ap


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "work/full_public_stage1"
RESULTS = ROOT / "outputs/full_public_stage1_results"
FIGS = RESULTS / "figures"
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
    FIGS.mkdir(parents=True, exist_ok=True)


def summarize_sequence(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, np.float32)
    diff = np.diff(arr, axis=0)
    if len(diff) == 0:
        diff = np.zeros_like(arr[:1])
    return np.concatenate(
        [
            arr.mean(0),
            arr.std(0),
            np.percentile(arr, 10, axis=0),
            np.percentile(arr, 50, axis=0),
            np.percentile(arr, 90, axis=0),
            np.abs(diff).mean(0),
        ]
    ).astype(np.float32)


def per_clip_mae(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y - pred), axis=(1, 2))


def per_clip_event_f1(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.asarray([ap.event_f1(y[i : i + 1], pred[i : i + 1]) for i in range(len(y))], dtype=np.float32)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ap.SeqDS, ap.SeqDS, ap.SeqDS]:
    feature_index = pd.read_csv(ap.CFG.table_dir / "real_feature_index.csv")
    train_df, val_df, test_df = ap.split_by_speaker(feature_index)
    train_ds = ap.SeqDS(train_df, fit=True)
    val_ds = ap.SeqDS(val_df, xs=train_ds.xs, ys=train_ds.ys)
    test_ds = ap.SeqDS(test_df, xs=train_ds.xs, ys=train_ds.ys)
    return train_df, val_df, test_df, train_ds, val_ds, test_ds


def run_stronger_simple_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame, train_ds: ap.SeqDS, test_ds: ap.SeqDS) -> pd.DataFrame:
    out_path = RESULTS / "strong_simple_baselines.csv"
    rows: list[dict] = []

    y = test_ds.Y_raw
    global_mean = train_ds.Y_raw.mean(axis=0, keepdims=True)
    pred = np.repeat(global_mean, len(test_ds), axis=0)
    rows.append({"model": "global_mean_motion_prior", "n_train": len(train_ds), "n_test": len(test_ds), **ap.metrics(y, pred)})

    dataset_means = {dataset: train_ds.Y_raw[train_df.dataset.to_numpy() == dataset].mean(axis=0) for dataset in sorted(train_df.dataset.unique())}
    pred = np.stack([dataset_means.get(row.dataset, global_mean[0]) for _, row in test_df.iterrows()]).astype(np.float32)
    rows.append({"model": "dataset_mean_motion_prior", "n_train": len(train_ds), "n_test": len(test_ds), **ap.metrics(y, pred)})

    feature_names = json.loads((ap.CFG.cache_root / "feature_names.json").read_text())
    rms_idx = feature_names.index("rms")
    train_energy = train_ds.Y_raw[:, :, -1].reshape(-1)
    mean_energy = float(train_energy.mean())
    std_energy = float(train_energy.std() + 1e-8)
    pred = np.repeat(global_mean, len(test_ds), axis=0)
    raw_x = [np.load(path).astype(np.float32) for path in test_df.feature_path]
    for idx, seq in enumerate(raw_x):
        rms = seq[:, rms_idx]
        rms = (rms - rms.mean()) / (rms.std() + 1e-8)
        pred[idx, :, -1] = np.maximum(0.0, mean_energy + 0.5 * std_energy * rms)
    rows.append({"model": "rms_energy_template_prior", "n_train": len(train_ds), "n_test": len(test_ds), **ap.metrics(y, pred)})

    train_summary = np.vstack([summarize_sequence(np.load(path)) for path in train_df.feature_path])
    test_summary = np.vstack([summarize_sequence(np.load(path)) for path in test_df.feature_path])
    scaler = StandardScaler().fit(train_summary)
    nearest, dist = pairwise_distances_argmin_min(scaler.transform(test_summary), scaler.transform(train_summary))
    pred = train_ds.Y_raw[nearest]
    rows.append(
        {
            "model": "nearest_audio_summary_retrieval",
            "n_train": len(train_ds),
            "n_test": len(test_ds),
            "mean_retrieval_distance": float(np.mean(dist)),
            **ap.metrics(y, pred),
        }
    )

    out = pd.DataFrame(rows).sort_values("overall_mae")
    out.to_csv(out_path, index=False)
    print(out[["model", "overall_mae", "motion_event_f1", "motion_energy_mae"]].to_string(index=False), flush=True)
    return out


def load_model(name: str, state_file: str, inp: int) -> torch.nn.Module:
    model = ap.make_model(name, inp, len(ap.TARGET_DIMS))
    model.load_state_dict(torch.load(ap.CFG.model_dir / state_file, map_location="cpu"))
    return model


def eval_model_on_x(model: torch.nn.Module, ds: ap.SeqDS, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    old = ds.X
    ds.X = x.astype(np.float32)
    try:
        return ap.pred_torch(model, ds)
    finally:
        ds.X = old


def run_mechanism_tests(test_df: pd.DataFrame, train_ds: ap.SeqDS, test_ds: ap.SeqDS) -> pd.DataFrame:
    out_path = RESULTS / "mechanism_stress_tests.csv"
    feature_names = json.loads((ap.CFG.cache_root / "feature_names.json").read_text())
    groups = {
        "clean": None,
        "prosody_energy_only": ["rms", "flux", "zcr", "plosive_proxy", "level_drift", "silence_prob", "silence_texture"],
        "radiation_geometry_only": ["hf_ratio", "mid_high_ratio", "centroid", "rolloff85", "rolloff95", "drr_proxy", "off_axis_proxy", "hf_drift", "centroid_drift"],
        "no_radiation_geometry": [name for name in feature_names if name not in {"hf_ratio", "mid_high_ratio", "centroid", "rolloff85", "rolloff95", "drr_proxy", "off_axis_proxy", "hf_drift", "centroid_drift"}],
    }

    rng = np.random.default_rng(SEED)
    conditions: dict[str, np.ndarray] = {}
    for condition, keep_names in groups.items():
        x = test_ds.X.copy()
        if keep_names is not None:
            keep = {feature_names.index(name) for name in keep_names if name in feature_names}
            mask = np.ones(x.shape[-1], dtype=bool)
            for idx in keep:
                mask[idx] = False
            x[:, :, mask] = 0.0
        conditions[condition] = x

    x = test_ds.X.copy()
    for idx in range(len(x)):
        x[idx] = x[idx, rng.permutation(x.shape[1])]
    conditions["within_clip_temporal_shuffle"] = x

    x = test_ds.X.copy()
    for dataset in sorted(test_df.dataset.unique()):
        ids = np.flatnonzero(test_df.dataset.to_numpy() == dataset)
        if len(ids) > 1:
            x[ids] = x[rng.permutation(ids)]
    conditions["within_dataset_clip_shuffle"] = x

    models = {
        "AcousticPose-FullTCN": load_model("acousticpose", "fulltrain_acousticpose.pt", test_ds.X.shape[-1]),
        "AcousticPose-Transformer++": load_model("transformer", "fulltrain_transformer.pt", test_ds.X.shape[-1]),
    }
    rows: list[dict] = []
    for model_name, model in models.items():
        for condition, x in conditions.items():
            print("mechanism", model_name, condition, flush=True)
            y, pred = eval_model_on_x(model, test_ds, x)
            rows.append({"model": model_name, "condition": condition, "n_clips": len(test_ds), **ap.metrics(y, pred)})
            pd.DataFrame(rows).to_csv(out_path, index=False)
    out = pd.DataFrame(rows).sort_values(["model", "overall_mae"])
    out.to_csv(out_path, index=False)
    return out


def facebox_proxy(video_path: str | Path, target_len: int = 160) -> tuple[np.ndarray, float]:
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(video_path))
    rows = []
    misses = 0
    total = 0
    last = None
    max_frames = int(ap.CFG.max_clip_sec * (cap.get(cv2.CAP_PROP_FPS) or ap.CFG.fps))
    for _ in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(35, 35))
        if len(faces):
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            cx = (x + 0.5 * w) / 320.0
            cy = (y + 0.5 * h) / 240.0
            scale = np.sqrt(w * h) / np.sqrt(320.0 * 240.0)
            aspect = w / (h + 1e-6)
            vec = np.asarray([cx - 0.5, cy - 0.5, aspect - 1.0, scale], np.float32)
        else:
            misses += 1
            vec = last.copy() if last is not None else np.zeros(4, np.float32)
        energy = 0.0 if last is None else float(np.linalg.norm(vec - last))
        rows.append([vec[0], vec[1], vec[2], vec[3], energy])
        last = vec.copy()
    cap.release()
    miss_rate = misses / max(total, 1)
    if not rows:
        return np.zeros((target_len, 5), np.float32), 1.0
    return ap.resize_seq(np.asarray(rows, np.float32), target_len), float(miss_rate)


def run_proxy_consistency(feature_index: pd.DataFrame, per_dataset: int = 24) -> pd.DataFrame:
    out_path = RESULTS / "raw_video_proxy_consistency.csv"
    rows: list[dict] = []
    sample = (
        feature_index.groupby("dataset", group_keys=False)
        .apply(lambda df: df.sample(min(per_dataset, len(df)), random_state=SEED))
        .reset_index(drop=True)
    )
    for _, row in sample.iterrows():
        print("proxy consistency", row.dataset, row.clip_id, flush=True)
        cached = np.load(row.target_path).astype(np.float32)
        facebox, miss_rate = facebox_proxy(row.video_path, target_len=cached.shape[0])
        for idx, name in enumerate(ap.TARGET_DIMS):
            a = cached[:, idx]
            b = facebox[:, idx]
            pearson = 0.0 if np.std(a) < 1e-8 or np.std(b) < 1e-8 else float(np.corrcoef(a, b)[0, 1])
            spearman = float(stats.spearmanr(a, b).correlation) if np.std(a) >= 1e-8 and np.std(b) >= 1e-8 else 0.0
            rows.append(
                {
                    "dataset": row.dataset,
                    "clip_id": row.clip_id,
                    "channel": name,
                    "pearson_r": pearson,
                    "spearman_rho": 0.0 if not np.isfinite(spearman) else spearman,
                    "facebox_miss_rate": miss_rate,
                }
            )
        pd.DataFrame(rows).to_csv(out_path, index=False)
    out = pd.DataFrame(rows)
    summary = (
        out.groupby(["dataset", "channel"])
        .agg(pearson_r=("pearson_r", "median"), spearman_rho=("spearman_rho", "median"), facebox_miss_rate=("facebox_miss_rate", "mean"))
        .reset_index()
    )
    summary.to_csv(RESULTS / "raw_video_proxy_consistency_summary.csv", index=False)
    return summary


def run_qualitative_cases(train_ds: ap.SeqDS, test_df: pd.DataFrame, test_ds: ap.SeqDS) -> pd.DataFrame:
    out_path = RESULTS / "deterministic_qualitative_cases.csv"
    ridge = Ridge(alpha=1.0)
    x_train, y_train = ap.flat_sample(train_ds, 300_000)
    ridge.fit(x_train, y_train)
    ridge_pred = test_ds.inverse_y(ridge.predict(ap.flat(test_ds)[0]).reshape(test_ds.Y.shape).astype(np.float32))
    transformer = load_model("transformer", "fulltrain_transformer.pt", test_ds.X.shape[-1])
    acousticpose = load_model("acousticpose", "fulltrain_acousticpose.pt", test_ds.X.shape[-1])
    y, trans_pred = ap.pred_torch(transformer, test_ds)
    _, ap_pred = ap.pred_torch(acousticpose, test_ds)

    mae = per_clip_mae(y, ap_pred)
    f1 = per_clip_event_f1(y, trans_pred)
    dataset = test_df.dataset.to_numpy()
    choices = {
        "median_acousticpose_mae": int(np.argsort(mae)[len(mae) // 2]),
        "best_transformer_event_f1": int(np.argmax(f1)),
        "worst_acousticpose_mae": int(np.argmax(mae)),
    }
    for ds_name, label, order in [
        ("MELD", "meld_failure_high_mae", -1),
        ("RAVDESS", "ravdess_success_low_mae", 0),
        ("CREMA-D", "cremad_success_low_mae", 0),
        ("CREMA-D", "cremad_failure_high_mae", -1),
    ]:
        ids = np.flatnonzero(dataset == ds_name)
        if len(ids):
            sorted_ids = ids[np.argsort(mae[ids])]
            choices[label] = int(sorted_ids[order])

    rows = []
    for label, idx in choices.items():
        rows.append(
            {
                "case": label,
                "dataset": test_df.iloc[idx].dataset,
                "clip_id": test_df.iloc[idx].clip_id,
                "acousticpose_mae": float(mae[idx]),
                "transformer_event_f1": float(f1[idx]),
                "ridge_mae": float(per_clip_mae(y[idx : idx + 1], ridge_pred[idx : idx + 1])[0]),
                "transformer_mae": float(per_clip_mae(y[idx : idx + 1], trans_pred[idx : idx + 1])[0]),
            }
        )
        raw_x = np.load(test_df.iloc[idx].feature_path)
        rms_idx = json.loads((ap.CFG.cache_root / "feature_names.json").read_text()).index("rms")
        rms = raw_x[:, rms_idx]
        rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
        t = np.arange(y.shape[1])
        plt.figure(figsize=(10, 4.5))
        plt.plot(t, rms * max(1e-6, y[idx, :, -1].max()), label="audio RMS (scaled)", linewidth=1.2)
        plt.plot(t, y[idx, :, -1], label="target motion energy", linewidth=1.6)
        plt.plot(t, ridge_pred[idx, :, -1], label="ridge", alpha=0.8)
        plt.plot(t, ap_pred[idx, :, -1], label="AcousticPose-FullTCN", alpha=0.9)
        plt.plot(t, trans_pred[idx, :, -1], label="AcousticPose-Transformer++", alpha=0.9)
        plt.title(f"{label}: {test_df.iloc[idx].dataset}/{test_df.iloc[idx].clip_id}")
        plt.xlabel("resampled frame")
        plt.ylabel("motion-energy proxy")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGS / f"qualitative_{label}.png", dpi=220)
        plt.close()

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def run_repeated_shallow_seeds(feature_index: pd.DataFrame) -> pd.DataFrame:
    out_path = RESULTS / "repeated_seed_shallow_results.csv"
    rows = []
    for seed in [11, 42, 123]:
        old_seed = ap.SEED
        ap.SEED = seed
        train_df, _, test_df = ap.split_by_speaker(feature_index)
        train_df = train_df.sample(min(3000, len(train_df)), random_state=seed).reset_index(drop=True)
        train_ds = ap.SeqDS(train_df, fit=True)
        test_ds = ap.SeqDS(test_df, xs=train_ds.xs, ys=train_ds.ys)
        ridge = Ridge(alpha=1.0)
        x_train, y_train = ap.flat_sample(train_ds, 160_000)
        ridge.fit(x_train, y_train)
        pred = test_ds.inverse_y(ridge.predict(ap.flat(test_ds)[0]).reshape(test_ds.Y.shape).astype(np.float32))
        rows.append({"seed": seed, "model": "ridge_frame", "train_clips_used": len(train_df), "test_clips_used": len(test_df), **ap.metrics(test_ds.Y_raw, pred)})
        ap.SEED = old_seed
        pd.DataFrame(rows).to_csv(out_path, index=False)
    out = pd.DataFrame(rows)
    summary = out.groupby("model").agg(overall_mae_mean=("overall_mae", "mean"), overall_mae_std=("overall_mae", "std"), event_f1_mean=("motion_event_f1", "mean"), event_f1_std=("motion_event_f1", "std")).reset_index()
    summary.to_csv(RESULTS / "repeated_seed_shallow_summary.csv", index=False)
    return out


def write_sota_coverage_table() -> pd.DataFrame:
    rows = [
        ("Gesticulator", "speech/text to 3D gesture", "Trinity/GENEA-style", "generation metrics", "No shared proxy predictions"),
        ("CaMN/BEAT baseline", "audio/text/emotion/speaker to gesture", "BEAT", "FGD/BC/L1/diversity", "Requires BEAT motion export"),
        ("TalkSHOW", "speech to holistic mesh/body", "TalkSHOW/BEAT-style", "FGD/LVD/diversity/user study", "No local checkpoint outputs"),
        ("EMAGE", "masked audio-gesture holistic generation", "BEAT2", "FGD/BC/MSE/LVD", "Different dataset and target space"),
        ("SemGes", "semantic-aware co-speech gesture", "BEAT2", "semantic coherence and generation metrics", "Needs text/audio/motion export"),
        ("GestureLSM/SemTalk", "latent/semantic holistic co-speech motion", "BEAT2", "FGD/BC/diversity/MSE", "Coverage comparison only"),
        ("SyncAnimation", "real-time audio-driven pose/talking head", "avatar video datasets", "synchrony/rendering metrics", "No comparable proxy labels"),
        ("EMO2/Audio2Photoreal-style", "audio-driven avatar video with end-effectors", "avatar video datasets", "visual/sync quality", "No proxy predictions mounted"),
        ("WavLM/Whisper encoder baseline", "self-supervised audio encoder plus temporal head", "our split if implemented", "MAE/F1", "Recommended next numeric baseline"),
    ]
    out = pd.DataFrame(rows, columns=["method_family", "task", "reported_data", "native_metrics", "status_for_this_paper"])
    out.to_csv(RESULTS / "external_sota_coverage_matrix.csv", index=False)
    return out


def main() -> None:
    configure()
    feature_index = pd.read_csv(ap.CFG.table_dir / "real_feature_index.csv")
    train_df, _, test_df, train_ds, _, test_ds = load_splits()
    print("extension split", len(train_df), len(test_df), feature_index.dataset.value_counts().to_dict(), flush=True)
    run_stronger_simple_baselines(train_df, test_df, train_ds, test_ds)
    run_mechanism_tests(test_df, train_ds, test_ds)
    run_proxy_consistency(feature_index)
    run_qualitative_cases(train_ds, test_df, test_ds)
    run_repeated_shallow_seeds(feature_index)
    write_sota_coverage_table()


if __name__ == "__main__":
    main()
