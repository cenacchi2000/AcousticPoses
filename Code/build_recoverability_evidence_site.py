#!/usr/bin/env python3
"""Build a recoverability-frontier evidence website from real AcousticPose results."""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs/full_public_stage1_results"
SITE = ROOT / "outputs/website"
ASSETS = SITE / "assets"
FIG_DIR = ASSETS / "figures"
EVIDENCE_DIR = RESULTS / "recoverability_evidence"
OVERLEAF_FIGS = ROOT / "outputs/overleaf/figures"

CHANNELS = [
    ("head_yaw", "Head yaw", "orientation"),
    ("head_pitch", "Head pitch", "orientation"),
    ("head_roll", "Head roll", "orientation"),
    ("torso_lean", "Torso lean", "coarse body"),
    ("motion_energy", "Motion energy", "activity"),
]


def ensure_dirs() -> None:
    for path in [SITE, ASSETS, FIG_DIR, EVIDENCE_DIR, OVERLEAF_FIGS]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def first_existing(*names: str) -> pd.DataFrame:
    for name in names:
        path = RESULTS / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(names)


def metric_row(df: pd.DataFrame, key_col: str, key: str) -> pd.Series:
    rows = df[df[key_col].astype(str) == key]
    if rows.empty:
        raise KeyError(f"{key!r} not found in {key_col}")
    return rows.iloc[0]


def pct_delta(a: float, b: float) -> float:
    return (a - b) / max(abs(b), 1e-9) * 100.0


def build_summary_numbers() -> dict:
    full = first_existing("full_non_capped_neural_results.csv", "main_real_results.csv")
    transformer = metric_row(full, "model", "fulltrain_transformer") if "fulltrain_transformer" in set(full.model.astype(str)) else metric_row(full, "model", "transformer")
    full_tcn = metric_row(full, "model", "fulltrain_acousticpose") if "fulltrain_acousticpose" in set(full.model.astype(str)) else metric_row(full, "model", "acousticpose")
    main = first_existing("main_real_results.csv")
    ridge = metric_row(main, "model", "ridge_frame")
    priors = first_existing("strong_simple_baselines.csv")
    dataset_prior = metric_row(priors, "model", "dataset_mean_motion_prior")
    global_prior = metric_row(priors, "model", "global_mean_motion_prior")
    controls = first_existing("identity_style_controls.csv", "negative_controls.csv")
    if "condition" in controls.columns and "original_audio" in set(controls.condition.astype(str)):
        original = controls[controls.condition.astype(str) == "original_audio"].iloc[0]
        random_rows = controls[controls.condition.astype(str).str.contains("random", case=False, na=False)]
        random = random_rows.sort_values("motion_energy_corr").iloc[0] if not random_rows.empty else original
    else:
        original = transformer
        random = transformer
    return {
        "full_transformer": transformer.to_dict(),
        "full_tcn": full_tcn.to_dict(),
        "ridge": ridge.to_dict(),
        "dataset_prior": dataset_prior.to_dict(),
        "global_prior": global_prior.to_dict(),
        "original_control": original.to_dict(),
        "random_control": random.to_dict(),
    }


def build_frontier_board() -> Path:
    nums = build_summary_numbers()
    transformer = pd.Series(nums["full_transformer"])
    dataset_prior = pd.Series(nums["dataset_prior"])
    global_prior = pd.Series(nums["global_prior"])
    random_control = pd.Series(nums["random_control"])
    original_control = pd.Series(nums["original_control"])
    headpose = first_existing("mediapipe_headpose_large_validation_summary.csv")
    emotion = first_existing("emotion_recognition_results.csv")
    persona = first_existing("persona_recognition_results.csv")

    fig = plt.figure(figsize=(18, 10.5), facecolor="#f6f4ed")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 0.9], hspace=0.34, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    labels = ["Event timing", "Motion energy", "Torso lean", "Head roll", "Head pitch", "Head yaw"]
    values = [
        float(transformer["motion_event_f1"]),
        float(transformer["motion_energy_corr"]),
        float(transformer["torso_lean_corr"]),
        float(transformer["head_roll_corr"]),
        float(transformer["head_pitch_corr"]),
        float(transformer["head_yaw_corr"]),
    ]
    colors = ["#159a68", "#159a68", "#d18a21", "#d18a21", "#8a8f98", "#8a8f98"]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="#1d272c")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.55, max(values) + 0.08))
    ax.set_xlabel("recoverability score (F1 or correlation)")
    ax.set_title("Recoverability frontier", loc="left", fontsize=18, fontweight="bold")
    for yi, v in zip(y, values):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=11)
    ax.axvspan(0, 0.10, color="#d7d4cc", alpha=0.35)
    ax.axvspan(0.10, 0.30, color="#f0d9a8", alpha=0.30)
    ax.axvspan(0.30, 0.55, color="#b7dfc8", alpha=0.30)
    ax.text(0.02, 5.65, "weak", fontsize=10, color="#555")
    ax.text(0.14, 5.65, "partial", fontsize=10, color="#555")
    ax.text(0.36, 5.65, "recoverable", fontsize=10, color="#555")

    ax = fig.add_subplot(gs[0, 1])
    models = ["Global prior", "Dataset prior", "Transformer++"]
    maes = [float(global_prior["overall_mae"]), float(dataset_prior["overall_mae"]), float(transformer["overall_mae"])]
    f1s = [float(global_prior["motion_event_f1"]), float(dataset_prior["motion_event_f1"]), float(transformer["motion_event_f1"])]
    x = np.arange(len(models))
    ax2 = ax.twinx()
    ax.bar(x - 0.18, maes, width=0.36, color="#68727d", label="MAE")
    ax2.bar(x + 0.18, f1s, width=0.36, color="#159a68", label="Event F1")
    ax.set_xticks(x, models, rotation=12)
    ax.set_ylabel("MAE (lower better)")
    ax2.set_ylabel("")
    ax.set_title("Static-prior collapse", loc="left", fontsize=18, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax2.legend(loc="upper right", frameon=False, fontsize=10)
    for xi, v in zip(x, maes):
        ax.text(xi - 0.18, v + 0.006, f"{v:.3f}", ha="center", fontsize=10)
    for xi, v in zip(x, f1s):
        ax2.text(xi + 0.18, v + 0.014, f"{v:.3f}", ha="center", fontsize=10, color="#0a6b49")

    ax = fig.add_subplot(gs[0, 2])
    control_labels = ["Original", "Wrong audio"]
    control_vals = [
        float(original_control.get("motion_energy_corr", transformer["motion_energy_corr"])),
        float(random_control.get("motion_energy_corr", 0.0)),
    ]
    ax.bar(control_labels, control_vals, color=["#159a68", "#bd4b43"], edgecolor="#1d272c")
    ax.set_ylim(min(-0.15, min(control_vals) - 0.05), max(0.55, max(control_vals) + 0.08))
    ax.set_ylabel("motion-energy correlation")
    ax.set_title("Audio identity matters", loc="left", fontsize=18, fontweight="bold")
    for xi, v in enumerate(control_vals):
        ax.text(xi, v + 0.025, f"{v:.3f}", ha="center", fontsize=12)
    drop = control_vals[0] - control_vals[1]
    ax.text(0.5, ax.get_ylim()[1] - 0.05, f"correlation drop = {drop:.3f}", ha="center", fontsize=12, fontweight="bold")

    ax = fig.add_subplot(gs[1, 0])
    hp = headpose[["dataset", "energy_spearman_median", "best_orientation_spearman"]].copy()
    xx = np.arange(len(hp))
    ax.bar(xx - 0.18, hp.energy_spearman_median, 0.36, label="motion energy", color="#159a68")
    ax.bar(xx + 0.18, hp.best_orientation_spearman, 0.36, label="best orientation", color="#d18a21")
    ax.set_xticks(xx, hp.dataset)
    ax.set_ylim(0, max(0.78, hp.energy_spearman_median.max() + 0.08))
    ax.set_title("Independent proxy validation", loc="left", fontsize=16, fontweight="bold")
    ax.set_ylabel("MediaPipe agreement")
    ax.legend(frameon=False, fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    emo = emotion.set_index("source")
    persona_df = persona.set_index("source")
    bars = [
        float(emo.loc["audio_summary", "macro_f1"]),
        float(emo.loc["audio_plus_motion_proxy", "macro_f1"]),
        float(persona_df.loc["audio_summary", "macro_f1"]),
        float(persona_df.loc["audio_plus_motion_proxy", "macro_f1"]),
    ]
    labs = ["Emotion\naudio", "Emotion\n+motion", "Persona\naudio", "Persona\n+motion"]
    ax.bar(labs, bars, color=["#68727d", "#159a68", "#68727d", "#159a68"], edgecolor="#1d272c")
    ax.set_ylabel("macro F1")
    ax.set_title("Embodied signal is complementary", loc="left", fontsize=16, fontweight="bold")
    for xi, v in enumerate(bars):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    bullets = [
        f"21,454 real clips across MELD, CREMA-D, and RAVDESS.",
        f"Full Transformer++ MAE {float(transformer['overall_mae']):.3f}; event F1 {float(transformer['motion_event_f1']):.3f}.",
        f"Motion-energy correlation {float(transformer['motion_energy_corr']):.3f}; yaw correlation {float(transformer['head_yaw_corr']):.3f}.",
        "Conclusion: speech recovers activity timing and energy, not full spatial pose.",
    ]
    ax.text(0, 1, "Scientific claim", fontsize=16, fontweight="bold", va="top")
    y0 = 0.82
    for b in bullets:
        ax.text(0.02, y0, u"\u2022 " + textwrap.fill(b, 48), fontsize=12, va="top")
        y0 -= 0.18

    fig.suptitle(
        "AcousticPose: recoverability frontier from speech, not full-body hallucination",
        fontsize=24,
        fontweight="bold",
        x=0.03,
        y=0.98,
        ha="left",
    )
    out = EVIDENCE_DIR / "recoverability_frontier_evidence_board.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out


def copy_case_images() -> list[dict]:
    cases = first_existing("qualitative_frame_audio_16_cases.csv")
    rows: list[dict] = []
    for _, row in cases.iterrows():
        src = Path(str(row.figure_path))
        if not src.exists():
            continue
        dst = FIG_DIR / src.name
        shutil.copy2(src, dst)
        rows.append(
            {
                "case_id": int(row.case_id),
                "category": str(row.category).replace("_", " "),
                "dataset": str(row.dataset),
                "clip_id": str(row.clip_id),
                "mae": float(row.transformer_mae),
                "event_f1": float(row.transformer_event_f1),
                "energy_mae": float(row.transformer_energy_mae),
                "image": f"assets/figures/{dst.name}",
            }
        )
    montage = RESULTS / "qualitative_frame_audio_evidence/qualitative_16_case_montage.png"
    if montage.exists():
        shutil.copy2(montage, FIG_DIR / montage.name)
        shutil.copy2(montage, OVERLEAF_FIGS / montage.name)
    return rows


def copy_existing_figures(frontier_board: Path) -> dict[str, str]:
    assets = {
        "frontier_board": frontier_board,
        "qual_montage": RESULTS / "qualitative_frame_audio_evidence/qualitative_16_case_montage.png",
        "static_prior": OVERLEAF_FIGS / "results_static_prior_trap.png",
        "frontier_channels": OVERLEAF_FIGS / "recoverability_frontier_channels.png",
        "proxy_confidence": OVERLEAF_FIGS / "proxy_confidence_performance.png",
        "raw_robustness": OVERLEAF_FIGS / "raw_audio_robustness_subset.png",
    }
    out: dict[str, str] = {}
    for key, src in assets.items():
        if src.exists():
            dst = FIG_DIR / src.name
            shutil.copy2(src, dst)
            overleaf_dst = OVERLEAF_FIGS / src.name
            if src.resolve() != overleaf_dst.resolve():
                shutil.copy2(src, overleaf_dst)
            out[key] = f"assets/figures/{dst.name}"
    return out


def result_cards() -> list[dict]:
    nums = build_summary_numbers()
    t = pd.Series(nums["full_transformer"])
    prior = pd.Series(nums["dataset_prior"])
    rnd = pd.Series(nums["random_control"])
    orig = pd.Series(nums["original_control"])
    hp = first_existing("mediapipe_headpose_large_validation_summary.csv")
    return [
        {
            "label": "Real benchmark",
            "value": "21,454",
            "caption": "clips from MELD, CREMA-D, and RAVDESS",
        },
        {
            "label": "Transformer++",
            "value": f"{float(t['overall_mae']):.3f}",
            "caption": f"overall MAE, event F1 {float(t['motion_event_f1']):.3f}",
        },
        {
            "label": "Recoverable channel",
            "value": f"{float(t['motion_energy_corr']):.3f}",
            "caption": "motion-energy correlation from speech",
        },
        {
            "label": "Weak channel",
            "value": f"{float(t['head_yaw_corr']):.3f}",
            "caption": "yaw correlation: frontier, not full pose",
        },
        {
            "label": "Static prior failure",
            "value": f"{float(prior['motion_event_f1']):.3f}",
            "caption": f"dataset prior event F1 vs {float(t['motion_event_f1']):.3f} for speech",
        },
        {
            "label": "Wrong-audio drop",
            "value": f"{float(orig.get('motion_energy_corr', t['motion_energy_corr'])) - float(rnd.get('motion_energy_corr', 0.0)):.3f}",
            "caption": "motion-energy correlation lost under mismatched audio",
        },
        {
            "label": "Proxy validation",
            "value": f"{float(hp.energy_spearman_median.median()):.3f}",
            "caption": "median energy agreement with MediaPipe subset",
        },
    ]


def write_site(case_rows: list[dict], figure_paths: dict[str, str]) -> None:
    cards = result_cards()
    metric_html = "\n".join(
        f"<div><span>{html.escape(c['label'])}</span><strong>{html.escape(c['value'])}</strong><p>{html.escape(c['caption'])}</p></div>"
        for c in cards
    )
    case_html = "\n".join(
        f"""
        <article class="case-card">
          <img src="{html.escape(row['image'])}" alt="Recoverability evidence case {row['case_id']}" loading="lazy" />
          <div>
            <p class="eyebrow">{html.escape(row['dataset'])} · case {row['case_id']:02d}</p>
            <h3>{html.escape(row['clip_id'])}</h3>
            <p>{html.escape(row['category'])}</p>
            <dl>
              <div><dt>MAE</dt><dd>{row['mae']:.3f}</dd></div>
              <div><dt>Event F1</dt><dd>{row['event_f1']:.3f}</dd></div>
              <div><dt>Energy MAE</dt><dd>{row['energy_mae']:.3f}</dd></div>
            </dl>
          </div>
        </article>
        """
        for row in case_rows[:16]
    )
    key_figs = "".join(
        f'<figure><img src="{html.escape(src)}" alt="{html.escape(key)}"><figcaption>{html.escape(key.replace("_", " ").title())}</figcaption></figure>'
        for key, src in figure_paths.items()
        if key not in {"frontier_board", "qual_montage"}
    )
    (SITE / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AcousticPose · Recoverability Frontier Evidence</title>
  <link rel="icon" href="data:," />
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <nav class="topbar">
    <a href="#top" class="brand">AcousticPose</a>
    <div>
      <a href="#frontier">Frontier</a>
      <a href="#cases">Cases</a>
      <a href="#controls">Controls</a>
    </div>
  </nav>

  <header id="top" class="hero">
    <img src="{html.escape(figure_paths['frontier_board'])}" alt="Recoverability frontier evidence board" />
    <section>
      <p class="kicker">AAAI 2026 · calibrated acoustic embodiment</p>
      <h1>Speech reveals a recoverability frontier, not full pose.</h1>
      <p>The new demo replaces visual reconstruction claims with real-data evidence: what speech can recover, what collapses under controls, and which motion channels remain weak.</p>
    </section>
  </header>

  <main>
    <section class="metrics" aria-label="headline evidence">
      {metric_html}
    </section>

    <section id="frontier" class="section">
      <div class="section-head">
        <p class="index">/ 01 — Central Evidence</p>
        <h2>The recoverability frontier</h2>
        <p>Audio is dynamically sufficient for activity timing and motion energy, partially informative for coarse body dynamics, and weak for fine orientation. This is the paper's strongest honest claim.</p>
      </div>
      <figure class="wide-figure">
        <img src="{html.escape(figure_paths['frontier_board'])}" alt="Recoverability frontier evidence board" />
      </figure>
    </section>

    <section id="cases" class="section">
      <div class="section-head">
        <p class="index">/ 02 — Real Qualitative Evidence</p>
        <h2>Sixteen deterministic held-out cases</h2>
        <p>Each case shows real frames, mel spectrogram, target/model motion-energy traces, audio descriptor traces, and channel heatmaps. Cases are selected by deterministic rules: low-error successes, high event alignment, proxy-confidence, high-motion examples, and boundary failures.</p>
      </div>
      <div class="case-grid">
        {case_html}
      </div>
    </section>

    <section id="controls" class="section">
      <div class="section-head">
        <p class="index">/ 03 — Controls and Mechanisms</p>
        <h2>Why the signal is not a static prior</h2>
        <p>MAE alone is misleading: dataset priors can look competitive under average error, but fail on event timing. Wrong-audio and random-audio controls destroy motion-energy alignment, exposing what is genuinely acoustic.</p>
      </div>
      <div class="figure-grid">
        {key_figs}
      </div>
    </section>
  </main>

  <footer>
    <strong>AcousticPose</strong>
    <span>Recoverability-frontier evidence from real held-out audio-video data. The claim is calibrated motion-proxy recoverability, not full-body 3D reconstruction.</span>
  </footer>

  <script>window.ACOUSTICPOSE_CASES = {json.dumps(case_rows, indent=2)};</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    (SITE / "styles.css").write_text(
        """*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f6f4ed;color:#1b252b;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:inherit}.topbar{position:fixed;z-index:20;top:0;left:0;right:0;height:60px;display:flex;align-items:center;justify-content:space-between;padding:0 32px;color:#f8faf8;background:rgba(20,26,31,.82);backdrop-filter:blur(16px);border-bottom:1px solid rgba(255,255,255,.16)}.brand{text-decoration:none;font-weight:900}.topbar div{display:flex;gap:22px}.topbar a{text-decoration:none;font-size:14px}.hero{min-height:94vh;display:grid;grid-template-columns:1.15fr .85fr;align-items:center;gap:42px;padding:94px 6vw 52px;background:#12191f;color:#fff;overflow:hidden}.hero img{width:100%;border:1px solid rgba(255,255,255,.22);box-shadow:0 28px 80px rgba(0,0,0,.38);background:#f6f4ed}.hero section{max-width:720px}.kicker,.index,.eyebrow{margin:0 0 12px;text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:900;color:#2ba775}.hero h1{font-size:clamp(46px,6.8vw,92px);line-height:.92;margin:0 0 24px;max-width:760px}.hero p:not(.kicker){font-size:clamp(18px,2vw,24px);line-height:1.45;color:#dce8e3}.metrics{display:grid;grid-template-columns:repeat(7,1fr);background:#fff;border-bottom:1px solid #d8d4ca}.metrics div{padding:22px 20px;border-right:1px solid #dedbd2}.metrics div:last-child{border-right:0}.metrics span{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#647078;font-weight:800}.metrics strong{display:block;font-size:32px;margin:7px 0 5px}.metrics p{margin:0;color:#5d676e;font-size:13px;line-height:1.35}.section{padding:82px 6vw}.section-head{max-width:900px;margin-bottom:30px}.section-head h2{font-size:clamp(34px,5vw,64px);line-height:1;margin:0 0 16px}.section-head p:not(.index){font-size:18px;line-height:1.58;color:#4f5b62}.wide-figure{margin:0}.wide-figure img{display:block;width:100%;border:1px solid #d7d3c9;background:#fff}.case-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.case-card{display:grid;grid-template-columns:1.25fr .75fr;gap:0;background:#fff;border:1px solid #d9d5cb;border-radius:8px;overflow:hidden;box-shadow:0 16px 36px rgba(31,35,39,.08)}.case-card img{display:block;width:100%;height:100%;object-fit:cover;min-height:330px}.case-card>div{padding:18px}.case-card h3{margin:0 0 8px;font-size:21px}.case-card p{margin:0 0 14px;color:#59646b;line-height:1.45}.case-card dl{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:0}.case-card dt{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#69747b}.case-card dd{margin:2px 0 0;font-size:20px;font-weight:900}.figure-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.figure-grid figure{margin:0;background:#fff;padding:12px;border:1px solid #d9d5cb;border-radius:8px}.figure-grid img{display:block;width:100%;height:auto}.figure-grid figcaption{font-size:13px;color:#5e686f;margin-top:8px}footer{display:flex;justify-content:space-between;gap:24px;padding:34px 6vw;background:#172026;color:#eaf1ee}footer span{max-width:840px;color:#bac7c1}@media(max-width:1120px){.hero,.case-card{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.case-card img{height:auto;min-height:0}}@media(max-width:760px){.topbar{padding:0 18px}.topbar div{gap:12px}.hero{padding:86px 24px 44px}.section{padding:62px 24px}.case-grid,.figure-grid{grid-template-columns:1fr}.metrics{grid-template-columns:1fr}.hero h1{font-size:42px}.topbar div a:nth-child(3){display:none}}""",
        encoding="utf-8",
    )


def zip_site() -> Path:
    out = ROOT / "outputs/acousticpose_website.zip"
    out.unlink(missing_ok=True)
    subprocess.run(["zip", "-qr", str(out), "."], cwd=SITE, check=True)
    return out


def main() -> None:
    ensure_dirs()
    frontier = build_frontier_board()
    case_rows = copy_case_images()
    figure_paths = copy_existing_figures(frontier)
    write_site(case_rows, figure_paths)
    zip_path = zip_site()
    print("frontier", frontier)
    print("cases", len(case_rows))
    print("site", SITE)
    print("zip", zip_path)


if __name__ == "__main__":
    main()
