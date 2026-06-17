#!/usr/bin/env python3
"""Generate rich qualitative frame/audio evidence sheets for real test samples."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from sklearn.linear_model import Ridge

import acousticpose_local as ap
from rigorous_evaluation_extensions import per_clip_event_f1, per_clip_mae


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "work/full_public_stage1"
RESULTS = ROOT / "outputs/full_public_stage1_results"
OUT_DIR = RESULTS / "qualitative_frame_audio_evidence"
OVERLEAF_FIGS = ROOT / "outputs/overleaf/figures"
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
    torch.set_num_threads(2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLEAF_FIGS.mkdir(parents=True, exist_ok=True)


def load_model(name: str, state_file: str, inp: int) -> torch.nn.Module:
    model = ap.make_model(name, inp, len(ap.TARGET_DIMS))
    model.load_state_dict(torch.load(ap.CFG.model_dir / state_file, map_location="cpu"))
    model.eval()
    return model


def norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-8)


def get_frames(video_path: str | Path, selected_proxy_frames: list[int], target_len: int = 160) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [np.zeros((120, 160, 3), np.uint8) for _ in selected_proxy_frames]
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or target_len)
    max_frames = max(1, min(count, int((cap.get(cv2.CAP_PROP_FPS) or ap.CFG.fps) * ap.CFG.max_clip_sec)))
    out = []
    for proxy_idx in selected_proxy_frames:
        frame_idx = int(np.clip(proxy_idx / max(target_len - 1, 1) * (max_frames - 1), 0, max_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((240, 320, 3), np.uint8)
        frame = cv2.cvtColor(cv2.resize(frame, (220, 150)), cv2.COLOR_BGR2RGB)
        out.append(frame)
    cap.release()
    return out


def select_frame_indices(target_energy: np.ndarray) -> list[int]:
    peaks, _ = find_peaks(target_energy, distance=12, prominence=np.std(target_energy) * 0.25 if np.std(target_energy) > 0 else 0.01)
    if len(peaks) >= 4:
        chosen = peaks[np.argsort(target_energy[peaks])[-4:]]
        return sorted(map(int, chosen))
    return [15, 55, 100, 145]


def fit_ridge(train_ds: ap.SeqDS) -> Ridge:
    ridge = Ridge(alpha=1.0)
    x_train, y_train = ap.flat_sample(train_ds, 150_000)
    ridge.fit(x_train, y_train)
    return ridge


def build_case_list(test_df: pd.DataFrame, y: np.ndarray, ap_pred: np.ndarray, tr_pred: np.ndarray) -> pd.DataFrame:
    mae = per_clip_mae(y, tr_pred)
    ap_mae = per_clip_mae(y, ap_pred)
    f1 = per_clip_event_f1(y, tr_pred)
    energy_mae = np.mean(np.abs(y[:, :, -1] - tr_pred[:, :, -1]), axis=1)
    target_energy = y[:, :, -1].mean(axis=1)
    dataset = test_df.dataset.to_numpy()
    rows = []
    used: set[int] = set()

    def add(label: str, indices: list[int], limit: int) -> None:
        added = 0
        for idx in indices:
            idx = int(idx)
            if idx in used:
                continue
            used.add(idx)
            rows.append(
                {
                    "case_id": len(rows) + 1,
                    "category": label,
                    "test_idx": idx,
                    "dataset": test_df.iloc[idx].dataset,
                    "clip_id": test_df.iloc[idx].clip_id,
                    "transformer_mae": float(mae[idx]),
                    "acousticpose_mae": float(ap_mae[idx]),
                    "transformer_event_f1": float(f1[idx]),
                    "transformer_energy_mae": float(energy_mae[idx]),
                    "target_energy_mean": float(target_energy[idx]),
                }
            )
            added += 1
            if added >= limit or len(rows) >= 16:
                break

    # Low-error successes, keeping dataset diversity.
    for ds in ["CREMA-D", "RAVDESS", "MELD"]:
        ids = np.flatnonzero(dataset == ds)
        add(f"success_low_mae_{ds}", ids[np.argsort(mae[ids])].tolist(), 2 if ds != "MELD" else 1)

    # High event-alignment examples.
    add("best_event_alignment", np.argsort(-f1).tolist(), 4)

    # High-confidence proxy examples from the independent face-box subset.
    conf_path = RESULTS / "proxy_confidence_detail.csv"
    if conf_path.exists():
        conf = pd.read_csv(conf_path).sort_values(["confidence_bin", "transformer_mae"], ascending=[True, True])
        high = conf[conf.confidence_bin.astype(str).isin(["high", "medium"])].sort_values("proxy_confidence", ascending=False)
        add("high_proxy_confidence", high.test_idx.astype(int).tolist(), 4)

    # High-motion examples are visually useful.
    add("high_motion_energy", np.argsort(-target_energy).tolist(), 3)

    # Boundary/failure cases expose limitations.
    add("failure_boundary_high_mae", np.argsort(-mae).tolist(), 4)

    # Fill any remaining slots deterministically.
    add("coverage_fill", np.argsort(mae).tolist(), 16 - len(rows))
    return pd.DataFrame(rows[:16])


def render_case(row: pd.Series, test_df: pd.DataFrame, y: np.ndarray, ridge_pred: np.ndarray, ap_pred: np.ndarray, tr_pred: np.ndarray, feature_names: list[str]) -> Path:
    idx = int(row.test_idx)
    meta = test_df.iloc[idx]
    raw_x = np.load(meta.feature_path).astype(np.float32)
    frames_idx = select_frame_indices(y[idx, :, -1])
    frames = get_frames(meta.video_path, frames_idx, target_len=y.shape[1])

    wav = ap.ensure_wav(meta.audio_path)
    audio, sr = librosa.load(str(wav), sr=ap.CFG.sr, mono=True, duration=ap.CFG.max_clip_sec)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, hop_length=max(1, int(sr / ap.CFG.fps)), n_fft=1024, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    t = np.arange(y.shape[1])
    fig = plt.figure(figsize=(14, 10), constrained_layout=False)
    gs = gridspec.GridSpec(5, 4, figure=fig, height_ratios=[1.15, 1.15, 1.2, 1.1, 1.0], hspace=0.65, wspace=0.18)

    for col, frame in enumerate(frames):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(frame)
        ax.set_title(f"frame {frames_idx[col]}", fontsize=9)
        ax.axis("off")

    ax_spec = fig.add_subplot(gs[1, :])
    librosa.display.specshow(mel_db, sr=sr, hop_length=max(1, int(sr / ap.CFG.fps)), x_axis="time", y_axis="mel", ax=ax_spec, cmap="magma")
    ax_spec.set_title("audio analysis: mel spectrogram", fontsize=10)
    ax_spec.set_xlabel("time (s)")

    ax_motion = fig.add_subplot(gs[2, :])
    ax_motion.plot(t, y[idx, :, -1], label="target motion energy", linewidth=2.0, color="black")
    ax_motion.plot(t, ridge_pred[idx, :, -1], label="ridge", alpha=0.75)
    ax_motion.plot(t, ap_pred[idx, :, -1], label="FullTCN", alpha=0.85)
    ax_motion.plot(t, tr_pred[idx, :, -1], label="Transformer++", alpha=0.95)
    for xline in frames_idx:
        ax_motion.axvline(xline, color="gray", linewidth=0.8, alpha=0.4)
    ax_motion.set_title("frame analysis: target/model motion-energy trajectories", fontsize=10)
    ax_motion.set_xlabel("resampled frame")
    ax_motion.set_ylabel("motion energy")
    ax_motion.legend(ncol=4, fontsize=8)

    ax_feat = fig.add_subplot(gs[3, :])
    for name in ["rms", "hf_ratio", "off_axis_proxy", "drr_proxy"]:
        if name in feature_names:
            vals = norm(raw_x[:, feature_names.index(name)])
            ax_feat.plot(t, vals, label=name, linewidth=1.2)
    ax_feat.set_title("audio descriptor traces used by AcousticPose", fontsize=10)
    ax_feat.set_xlabel("resampled frame")
    ax_feat.set_ylabel("normalized value")
    ax_feat.legend(ncol=4, fontsize=8)

    ax_heat = fig.add_subplot(gs[4, :])
    heat = np.vstack([norm(y[idx, :, d]) for d in range(y.shape[-1])])
    ax_heat.imshow(heat, aspect="auto", cmap="viridis", interpolation="nearest")
    ax_heat.set_yticks(range(len(ap.TARGET_DIMS)))
    ax_heat.set_yticklabels(ap.TARGET_DIMS, fontsize=8)
    ax_heat.set_xlabel("resampled frame")
    ax_heat.set_title("target proxy-channel heatmap", fontsize=10)

    fig.suptitle(
        f"Case {int(row.case_id):02d}: {row.category} | {meta.dataset}/{meta.clip_id} | "
        f"MAE={row.transformer_mae:.3f}, F1={row.transformer_event_f1:.3f}, energy MAE={row.transformer_energy_mae:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    out = OUT_DIR / f"case_{int(row.case_id):02d}_{meta.dataset}_{meta.clip_id}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def make_montage(case_paths: list[Path], cases: pd.DataFrame) -> Path:
    thumbs = []
    for path in case_paths:
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (360, 260))
        thumbs.append(img)
    canvas = np.ones((4 * 300, 4 * 390, 3), np.uint8) * 255
    for i, img in enumerate(thumbs[:16]):
        r, c = divmod(i, 4)
        y0, x0 = r * 300, c * 390
        canvas[y0 : y0 + 260, x0 : x0 + 360] = img
        label = f"{i+1:02d} {cases.iloc[i].dataset} {cases.iloc[i].category[:22]}"
        cv2.putText(canvas, label, (x0 + 8, y0 + 286), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    out = OUT_DIR / "qualitative_16_case_montage.png"
    cv2.imwrite(str(out), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return out


def main() -> None:
    configure()
    feature_index = pd.read_csv(ap.CFG.table_dir / "real_feature_index.csv")
    train_df, val_df, test_df = ap.split_by_speaker(feature_index)
    train_ds = ap.SeqDS(train_df, fit=True)
    test_ds = ap.SeqDS(test_df, xs=train_ds.xs, ys=train_ds.ys)
    feature_names = json.loads((ap.CFG.cache_root / "feature_names.json").read_text())

    ridge = fit_ridge(train_ds)
    ridge_pred = test_ds.inverse_y(ridge.predict(ap.flat(test_ds)[0]).reshape(test_ds.Y.shape).astype(np.float32))
    acousticpose = load_model("acousticpose", "fulltrain_acousticpose.pt", test_ds.X.shape[-1])
    transformer = load_model("transformer", "fulltrain_transformer.pt", test_ds.X.shape[-1])
    y, ap_pred = ap.pred_torch(acousticpose, test_ds)
    _, tr_pred = ap.pred_torch(transformer, test_ds)

    cases = build_case_list(test_df, y, ap_pred, tr_pred)
    paths = []
    for _, row in cases.iterrows():
        print("render", int(row.case_id), row.dataset, row.clip_id, row.category, flush=True)
        path = render_case(row, test_df, y, ridge_pred, ap_pred, tr_pred, feature_names)
        paths.append(path)
        cases.loc[cases.case_id == row.case_id, "figure_path"] = str(path)
    montage = make_montage(paths, cases)
    cases["montage_path"] = str(montage)
    cases.to_csv(RESULTS / "qualitative_frame_audio_16_cases.csv", index=False)

    # Copy montage and the first four strongest examples into Overleaf figures.
    overleaf_montage = OVERLEAF_FIGS / "qualitative_16_case_montage.png"
    overleaf_montage.write_bytes(montage.read_bytes())
    for path in paths[:4]:
        (OVERLEAF_FIGS / path.name).write_bytes(path.read_bytes())
    print("wrote", montage, flush=True)
    print(cases[["case_id", "category", "dataset", "clip_id", "transformer_mae", "transformer_event_f1", "figure_path"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
