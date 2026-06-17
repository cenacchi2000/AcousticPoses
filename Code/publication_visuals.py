#!/usr/bin/env python3
"""Create publication-quality AcousticPose figures from completed real-data results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs/full_public_stage1_results"
FIGS = ROOT / "outputs/overleaf/figures"


COLORS = {
    "ours": "#1B9E77",
    "ours2": "#0072B2",
    "baseline": "#6B7280",
    "prior": "#D55E00",
    "warn": "#B91C1C",
    "accent": "#7C3AED",
    "light": "#F3F4F6",
    "text": "#111827",
}


def setup() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 260,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    path = FIGS / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def pretty_label(label: str) -> str:
    return (
        label.replace("AcousticPose-", "")
        .replace("fulltrain_", "")
        .replace("_", " ")
        .replace("motion prior", "prior")
        .replace("Transformer++ full", "Transformer++")
        .replace("FullTCN full", "FullTCN")
    )


def static_prior_trap() -> None:
    simple = pd.read_csv(RESULTS / "strong_simple_baselines.csv")
    full = pd.read_csv(RESULTS / "full_non_capped_neural_results.csv")
    classic = pd.read_csv(RESULTS / "classic_audio_representation_baselines.csv")
    rows = []
    mapping = {
        "global_mean_motion_prior": "Global mean prior",
        "dataset_mean_motion_prior": "Dataset mean prior",
        "rms_energy_template_prior": "RMS template prior",
        "nearest_audio_summary_retrieval": "Audio retrieval",
    }
    for model, label in mapping.items():
        r = simple[simple.model == model].iloc[0]
        rows.append((label, r.overall_mae, r.motion_event_f1, r.motion_energy_mae, "prior"))
    r = classic[classic.representation == "MFCC+deltas ridge-frame"].iloc[0]
    rows.append(("MFCC+deltas ridge", r.overall_mae, r.motion_event_f1, r.motion_energy_mae, "baseline"))
    r = classic[classic.representation == "AcousticPose descriptors ridge-frame"].iloc[0]
    rows.append(("Descriptor ridge", r.overall_mae, r.motion_event_f1, r.motion_energy_mae, "baseline"))
    for model, label in [("fulltrain_acousticpose", "FullTCN"), ("fulltrain_transformer", "Transformer++")]:
        r = full[full.model == model].iloc[0]
        rows.append((label, r.overall_mae, r.motion_event_f1, r.motion_energy_mae, "ours"))
    df = pd.DataFrame(rows, columns=["method", "mae", "f1", "energy_mae", "kind"])
    colors = [COLORS["prior"] if k == "prior" else COLORS["baseline"] if k == "baseline" else COLORS["ours"] for k in df.kind]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1.0]})
    order = df.sort_values("mae")
    axes[0].barh(order.method, order.mae, color=[colors[df.index.get_loc(i)] for i in order.index], alpha=0.92)
    axes[0].axvline(0.173225, color=COLORS["prior"], lw=1.6, ls="--", label="dataset mean prior")
    axes[0].set_xlabel("Overall MAE (lower is better)")
    axes[0].set_title("MAE alone hides the static-prior trap")
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")

    axes[1].scatter(df.mae, df.f1, s=135, c=colors, edgecolor="white", linewidth=1.4)
    for _, r in df.iterrows():
        axes[1].annotate(r.method, (r.mae, r.f1), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("Overall MAE (lower is better)")
    axes[1].set_ylabel("Motion-event F1 (higher is better)")
    axes[1].set_title("AcousticPose preserves timing while priors collapse")
    axes[1].set_xlim(df.mae.min() - 0.01, df.mae.max() + 0.02)
    axes[1].set_ylim(0.05, 0.48)
    fig.suptitle("Core result: static averages can match MAE, but not temporal activity", fontsize=14, fontweight="bold")
    save(fig, "results_static_prior_trap.png")


def recoverability_frontier() -> None:
    main = pd.read_csv(RESULTS / "main_real_results.csv")
    ridge = main[main.model == "ridge_frame"].iloc[0]
    acoustic = main[main.model == "acousticpose"].iloc[0]
    full = pd.read_csv(RESULTS / "full_non_capped_neural_results.csv")
    trans = full[full.model == "fulltrain_transformer"].iloc[0]
    channels = ["head_yaw", "head_pitch", "head_roll", "torso_lean", "motion_energy"]
    labels = ["Head yaw", "Head pitch", "Head roll", "Torso lean", "Motion energy"]
    ridge_vals = np.array([ridge[f"{c}_mae"] for c in channels])
    trans_vals = np.array([trans[f"{c}_mae"] for c in channels])
    gain = (ridge_vals - trans_vals) / ridge_vals * 100
    proxy = pd.read_csv(RESULTS / "raw_video_proxy_consistency_summary.csv")
    reliability = []
    for c in channels:
        rows = proxy[(proxy.channel == c) & (proxy.dataset.isin(["CREMA-D", "RAVDESS"]))]
        reliability.append(float(rows.spearman_rho.median()) if len(rows) else 0.0)

    x = np.arange(len(channels))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.1, 0.9]})
    width = 0.36
    axes[0].bar(x - width / 2, ridge_vals, width, label="Ridge frame", color=COLORS["baseline"])
    axes[0].bar(x + width / 2, trans_vals, width, label="Transformer++", color=COLORS["ours"])
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Per-channel MAE")
    axes[0].set_title("Motion energy is the largest recoverable channel")
    axes[0].legend()

    axes[1].barh(labels, gain, color=[COLORS["ours"] if g > 0 else COLORS["warn"] for g in gain], alpha=0.9)
    axes[1].axvline(0, color="#111827", lw=1)
    for i, (g, rel) in enumerate(zip(gain, reliability)):
        if g < 0:
            axes[1].text(0.55, i, f"boundary {g:.1f}% | rho {rel:.2f}", va="center", ha="left", fontsize=8, color=COLORS["warn"])
        else:
            axes[1].text(g + 0.8, i, f"{g:.1f}% | rho {rel:.2f}", va="center", ha="left", fontsize=8)
    axes[1].set_xlabel("Relative MAE reduction vs ridge (%)")
    axes[1].set_title("Recoverability frontier, with proxy reliability")
    axes[1].invert_yaxis()
    axes[1].set_xlim(-1.5, max(32, gain.max() + 4))
    fig.suptitle("Recoverability frontier: timing/energy is primary, orientation is bounded", fontsize=14, fontweight="bold")
    save(fig, "recoverability_frontier_channels.png")


def mechanism_controls() -> None:
    mech = pd.read_csv(RESULTS / "mechanism_stress_tests.csv")
    neg = pd.read_csv(RESULTS / "negative_controls.csv")
    rows = []
    wanted = [
        ("AcousticPose-FullTCN", "clean", "FullTCN clean"),
        ("AcousticPose-FullTCN", "no_radiation_geometry", "FullTCN no rad/geom"),
        ("AcousticPose-FullTCN", "prosody_energy_only", "FullTCN prosody only"),
        ("AcousticPose-Transformer++", "clean", "Trans++ clean"),
        ("AcousticPose-Transformer++", "no_radiation_geometry", "Trans++ no rad/geom"),
    ]
    for model, cond, label in wanted:
        r = mech[(mech.model == model) & (mech.condition == cond)].iloc[0]
        rows.append((label, r.overall_mae, r.motion_energy_mae, r.motion_event_f1, "mask"))
    for cond, label in [("random_clip_audio", "Trans++ random audio"), ("rms_only_features", "Trans++ RMS only")]:
        r = neg[(neg.model == "AcousticPose-Transformer++") & (neg.condition == cond)].iloc[0]
        rows.append((label, r.overall_mae, r.motion_energy_mae, r.motion_event_f1, "negative"))
    df = pd.DataFrame(rows, columns=["condition", "mae", "energy", "f1", "kind"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    colors = [COLORS["ours"] if "clean" in c else COLORS["warn"] if k == "negative" else COLORS["prior"] for c, k in zip(df.condition, df.kind)]
    axes[0].barh(df.condition, df.energy, color=colors, alpha=0.9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Motion-energy MAE")
    axes[0].set_title("Mechanism removal damages energy recovery")
    axes[1].scatter(df.mae, df.f1, s=120, c=colors, edgecolor="white", linewidth=1.3)
    for _, r in df.iterrows():
        axes[1].annotate(r.condition, (r.mae, r.f1), xytext=(5, 3), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("Overall MAE")
    axes[1].set_ylabel("Event F1")
    axes[1].set_title("Controls separate clean recovery from degraded inputs")
    fig.suptitle("Mechanism and negative controls", fontsize=14, fontweight="bold")
    save(fig, "mechanism_negative_controls.png")


def proxy_confidence() -> None:
    conf = pd.read_csv(RESULTS / "proxy_confidence_bins.csv")
    order = ["low", "medium", "high"]
    conf["confidence_bin"] = pd.Categorical(conf.confidence_bin, categories=order, ordered=True)
    conf = conf.sort_values("confidence_bin")
    x = np.arange(len(conf))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    width = 0.35
    axes[0].bar(x - width / 2, conf.ridge_mae, width, label="Ridge", color=COLORS["baseline"])
    axes[0].bar(x + width / 2, conf.transformer_mae, width, label="Transformer++", color=COLORS["ours"])
    axes[0].set_xticks(x, conf.confidence_bin.astype(str).str.title())
    axes[0].set_ylabel("Overall MAE")
    axes[0].set_title("Low-confidence proxy clips are hardest")
    axes[0].legend()
    axes[1].plot(x, conf.motion_energy_spearman, marker="o", label="face-box energy Spearman", color=COLORS["ours2"], lw=2)
    axes[1].plot(x, conf.miss_rate, marker="o", label="detector miss rate", color=COLORS["warn"], lw=2)
    axes[1].set_xticks(x, conf.confidence_bin.astype(str).str.title())
    axes[1].set_ylabel("Rate / correlation")
    axes[1].set_title("Confidence definition exposes label quality")
    axes[1].legend()
    fig.suptitle("Proxy confidence explains the public-video boundary", fontsize=14, fontweight="bold")
    save(fig, "proxy_confidence_performance.png")


def raw_audio_robustness() -> None:
    source = RESULTS / "raw_audio_robustness_expanded_summary.csv"
    title = "Raw-audio corruption subset, 1,500 held-out clips"
    if not source.exists():
        source = RESULTS / "raw_audio_robustness_subset.csv"
        title = "Raw-audio corruption subset, 360 held-out clips"
    raw = pd.read_csv(source)
    raw = raw[raw.model == "AcousticPose-Transformer++"].copy()
    order = ["clean", "telephone_bandpass", "reverb_rt60_06", "reverb_rt60_10", "packet_loss_10", "gain_shift_6db", "noise_20db", "noise_10db"]
    raw["condition"] = pd.Categorical(raw.condition, categories=order, ordered=True)
    raw = raw.sort_values("condition")
    raw = raw[raw.condition.notna()]
    label_map = {
        "clean": "Clean",
        "telephone_bandpass": "Telephone",
        "reverb_rt60_06": "Reverb 0.6",
        "reverb_rt60_10": "Reverb 1.0",
        "packet_loss_10": "Packet loss",
        "gain_shift_6db": "Gain +6dB",
        "noise_20db": "Noise 20dB",
        "noise_10db": "Noise 10dB",
        "reverb_rt60_proxy": "Reverb",
    }
    labels = [label_map[str(c)] for c in raw.condition]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    colors = [COLORS["ours"] if d < 0.1 else COLORS["prior"] if d < 0.3 else COLORS["warn"] for d in raw.relative_mae_degradation]
    axes[0].bar(labels, raw.overall_mae, color=colors, alpha=0.92)
    axes[0].set_ylabel("Overall MAE")
    axes[0].set_title("Robustness is strong except additive noise")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, raw.motion_energy_mae, color=colors, alpha=0.92)
    axes[1].set_ylabel("Motion-energy MAE")
    axes[1].set_title("Noise corrupts energy descriptors before extraction")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    save(fig, "raw_audio_robustness_subset.png")


def per_dataset_summary() -> None:
    df = pd.read_csv(RESULTS / "per_dataset_method_results_compact.csv")
    methods = ["ridge_frame", "transformer", "drop_off_axis_proxy"]
    names = {"ridge_frame": "Ridge", "transformer": "Transformer++", "drop_off_axis_proxy": "Best pruned"}
    sub = df[df.model.isin(methods) & df.dataset.isin(["CREMA-D", "MELD", "RAVDESS"])].copy()
    datasets = ["CREMA-D", "RAVDESS", "MELD"]
    x = np.arange(len(datasets))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    width = 0.24
    for j, m in enumerate(methods):
        vals = [sub[(sub.dataset == ds) & (sub.model == m)].overall_mae.iloc[0] for ds in datasets]
        axes[0].bar(x + (j - 1) * width, vals, width, label=names[m], color=[COLORS["baseline"], COLORS["ours2"], COLORS["ours"]][j], alpha=0.9)
        evals = [sub[(sub.dataset == ds) & (sub.model == m)].motion_energy_mae.iloc[0] for ds in datasets]
        axes[1].bar(x + (j - 1) * width, evals, width, label=names[m], color=[COLORS["baseline"], COLORS["ours2"], COLORS["ours"]][j], alpha=0.9)
    for ax, title, ylabel in [(axes[0], "Overall MAE by dataset", "MAE"), (axes[1], "Motion-energy MAE by dataset", "Energy MAE")]:
        ax.set_xticks(x, datasets)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle("Generalization: acted datasets are clean; MELD is the boundary case", fontsize=14, fontweight="bold")
    save(fig, "per_dataset_generalization.png")


def qualitative_case_grid() -> None:
    cases = pd.read_csv(RESULTS / "qualitative_frame_audio_16_cases.csv")
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    order = cases.sort_values(["category", "transformer_mae"]).reset_index(drop=True)
    colors = [
        COLORS["ours"] if "success" in c else COLORS["ours2"] if "event" in c else COLORS["accent"] if "confidence" in c else COLORS["warn"]
        for c in order.category
    ]
    ax.bar(np.arange(len(order)), order.transformer_mae, color=colors, alpha=0.9)
    ax.set_xticks(np.arange(len(order)), [f"{int(r.case_id):02d}\\n{r.dataset}" for _, r in order.iterrows()], rotation=0)
    ax.set_ylabel("Transformer++ MAE")
    ax.set_title("Sixteen real qualitative evidence sheets: successes, event-alignment, confidence, and boundaries")
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(order)), order.transformer_event_f1, color="#111827", marker="o", lw=1.6, label="Event F1")
    ax2.set_ylabel("Event F1")
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper right")
    save(fig, "qualitative_16_case_metrics.png")


def all_figures() -> None:
    setup()
    # Architecture figure is hand-authored as a static SVG/PNG asset.
    # Do not regenerate it here; this script only refreshes result plots.
    static_prior_trap()
    recoverability_frontier()
    mechanism_controls()
    proxy_confidence()
    raw_audio_robustness()
    per_dataset_summary()
    qualitative_case_grid()


if __name__ == "__main__":
    all_figures()
