"""
Tracking Metrics Graph Generator
==================================
Generates presentation-ready PNG plots from the per-frame metrics collected
by ``MetricsCollector``.

Requires **matplotlib** (``pip install matplotlib``).  If matplotlib is not
installed the module prints a warning and silently skips graph generation.

Generated plots
---------------
1. ``humans_per_frame.png``     – Humans detected per frame (line + smoothed)
2. ``active_tracks.png``        – Active track count over time
3. ``confidence_dist.png``      – Detection confidence histogram
4. ``track_duration_hist.png``  – Track duration distribution
5. ``fps_over_time.png``        – FPS performance over time

All files are saved into the supplied output directory.
"""

import os
import json
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")              # headless backend – no GUI needed
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _smooth(y, window=15):
    """Simple moving-average smoother for line plots."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def generate_graphs(metrics_json_path, output_dir):
    """
    Read a metrics JSON file and render all graphs.

    Parameters
    ----------
    metrics_json_path : str
        Path to the JSON file produced by ``MetricsCollector.save_json()``.
    output_dir : str
        Directory to write PNG files into.
    """
    if not _HAS_MPL:
        print("[WARNING] matplotlib not installed – skipping graph generation. "
              "Run: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(metrics_json_path, "r") as f:
        data = json.load(f)

    per_frame = data["per_frame"]
    summary = data.get("summary", {})
    track_durations = data.get("track_durations", {})

    frames = [r["frame_id"] for r in per_frame]
    num_dets = [r["num_detections"] for r in per_frame]
    num_tracks = [r["num_active_tracks"] for r in per_frame]
    avg_confs = [r["avg_confidence"] for r in per_frame]
    fps_vals = [r["fps"] for r in per_frame]

    # Style
    plt.rcParams.update({
        "figure.figsize": (12, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 12,
    })

    # ----------------------------------------------------------------
    # 1. Humans detected per frame
    # ----------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.fill_between(frames, num_dets, alpha=0.25, color="steelblue")
    ax.plot(frames, num_dets, linewidth=0.6, alpha=0.5, color="steelblue", label="Raw")
    ax.plot(frames, _smooth(num_dets), linewidth=2, color="navy", label="Smoothed")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Humans Detected")
    ax.set_title("Humans Detected per Frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "humans_per_frame.png"), dpi=150)
    plt.close(fig)

    # ----------------------------------------------------------------
    # 2. Active tracks vs frame
    # ----------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.fill_between(frames, num_tracks, alpha=0.25, color="seagreen")
    ax.plot(frames, num_tracks, linewidth=0.6, alpha=0.5, color="seagreen", label="Raw")
    ax.plot(frames, _smooth(num_tracks), linewidth=2, color="darkgreen", label="Smoothed")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Active Tracks")
    ax.set_title("Active Tracks over Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "active_tracks.png"), dpi=150)
    plt.close(fig)

    # ----------------------------------------------------------------
    # 3. Detection confidence distribution
    # ----------------------------------------------------------------
    all_confs = [c for c in avg_confs if c > 0]
    if all_confs:
        fig, ax = plt.subplots()
        ax.hist(all_confs, bins=50, color="coral", edgecolor="darkred", alpha=0.8)
        ax.axvline(np.mean(all_confs), color="red", linestyle="--",
                   label=f"Mean = {np.mean(all_confs):.3f}")
        ax.axvline(np.median(all_confs), color="blue", linestyle="--",
                   label=f"Median = {np.median(all_confs):.3f}")
        ax.set_xlabel("Average Confidence")
        ax.set_ylabel("Frequency")
        ax.set_title("Detection Confidence Distribution (per-frame avg)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "confidence_dist.png"), dpi=150)
        plt.close(fig)

    # ----------------------------------------------------------------
    # 4. Track duration histogram
    # ----------------------------------------------------------------
    durations = list(track_durations.values()) if track_durations else []
    if durations:
        durations = [int(d) for d in durations]
        fig, ax = plt.subplots()
        max_dur = max(durations)
        bins = min(50, max_dur) if max_dur > 0 else 10
        ax.hist(durations, bins=bins, color="mediumpurple", edgecolor="indigo", alpha=0.8)
        ax.axvline(np.mean(durations), color="red", linestyle="--",
                   label=f"Mean = {np.mean(durations):.1f} frames")
        ax.axvline(np.median(durations), color="blue", linestyle="--",
                   label=f"Median = {np.median(durations):.1f} frames")
        ax.set_xlabel("Track Duration (frames)")
        ax.set_ylabel("Number of Tracks")
        ax.set_title("Track Duration Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "track_duration_hist.png"), dpi=150)
        plt.close(fig)

    # ----------------------------------------------------------------
    # 5. FPS over time
    # ----------------------------------------------------------------
    positive_fps = [(f, v) for f, v in zip(frames, fps_vals) if v > 0]
    if positive_fps:
        fps_frames, fps_values = zip(*positive_fps)
        fig, ax = plt.subplots()
        ax.plot(fps_frames, fps_values, linewidth=0.6, alpha=0.4, color="orange", label="Raw FPS")
        ax.plot(fps_frames, _smooth(list(fps_values)), linewidth=2, color="darkorange", label="Smoothed")
        ax.axhline(np.mean(fps_values), color="red", linestyle="--",
                   label=f"Mean = {np.mean(fps_values):.1f} FPS")
        ax.set_xlabel("Frame")
        ax.set_ylabel("FPS")
        ax.set_title("Processing Speed over Time")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fps_over_time.png"), dpi=150)
        plt.close(fig)

    # ----------------------------------------------------------------
    # 6. Summary text card
    # ----------------------------------------------------------------
    if summary:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        lines = []
        for k, v in summary.items():
            label = k.replace("_", " ").title()
            lines.append(f"{label}: {v}")
        text = "\n".join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                          edgecolor="gray"))
        ax.set_title("Tracking Summary", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "summary_card.png"), dpi=150)
        plt.close(fig)

    print(f"[INFO] Saved {6 if durations and all_confs else 4}+ graphs to {output_dir}/")
