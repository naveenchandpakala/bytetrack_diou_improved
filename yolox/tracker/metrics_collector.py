"""
Tracking Metrics Collector
===========================
Collects per-frame and aggregate statistics during a tracking run and
exports them as CSV + JSON for analysis and presentation.

Tracked metrics
---------------
- **Per-frame**: frame_id, num_detections, num_active_tracks, avg_confidence,
  fps, lost_count, occluded_count
- **Aggregate**: total_frames, total_unique_ids, total_id_switches,
  avg_fps, median_fps, avg_track_duration, max_track_duration,
  median_confidence

Usage
-----
::

    collector = MetricsCollector()
    # inside frame loop:
    collector.update(frame_id, online_tlwhs, online_ids, online_scores, fps)
    # after loop:
    collector.finalize()
    collector.save_csv("metrics.csv")
    collector.save_json("metrics.json")
"""

import csv
import json
import os
import numpy as np
from collections import defaultdict


class MetricsCollector:
    """Lightweight per-frame + aggregate metrics for MOT evaluation."""

    def __init__(self):
        # Per-frame records
        self.frame_records = []

        # Track-level bookkeeping
        self._prev_ids = set()           # IDs in previous frame
        self._all_ids = set()            # all IDs ever seen
        self._track_first_frame = {}     # track_id → first frame
        self._track_last_frame = {}      # track_id → last frame
        self._id_switch_count = 0

        # Raw score buffer (for aggregate statistics)
        self._all_scores = []

    def update(self, frame_id, tlwhs, track_ids, scores, fps,
               lost_count=0, occluded_count=0):
        """
        Record one frame of tracking output.

        Parameters
        ----------
        frame_id : int
        tlwhs : list
            Bounding boxes (only used for count).
        track_ids : list[int]
        scores : list[float]
        fps : float
        lost_count : int   (optional) number of lost tracks this frame
        occluded_count : int (optional)
        """
        num_dets = len(tlwhs)
        num_tracks = len(track_ids)
        avg_conf = float(np.mean(scores)) if len(scores) > 0 else 0.0

        # --- ID switch detection -----------------------------------------
        current_ids = set(int(i) for i in track_ids)

        # An ID switch happens when an ID present in the previous frame
        # disappears AND a brand-new ID appears in its place within the
        # same frame.  This is a simplified heuristic; the official
        # metric uses spatial overlap, but this gives a good estimate.
        disappeared = self._prev_ids - current_ids
        appeared_new = current_ids - self._all_ids
        # Coarse estimate: min of disappeared & new-appeared counts
        frame_switches = min(len(disappeared), len(appeared_new))
        self._id_switch_count += frame_switches

        self._prev_ids = current_ids
        self._all_ids.update(current_ids)

        for tid in track_ids:
            tid = int(tid)
            if tid not in self._track_first_frame:
                self._track_first_frame[tid] = frame_id
            self._track_last_frame[tid] = frame_id

        self._all_scores.extend([float(s) for s in scores])

        record = {
            "frame_id": int(frame_id),
            "num_detections": num_dets,
            "num_active_tracks": num_tracks,
            "avg_confidence": round(avg_conf, 4),
            "fps": round(float(fps), 2),
            "id_switches_cumulative": self._id_switch_count,
            "lost_count": lost_count,
            "occluded_count": occluded_count,
        }
        self.frame_records.append(record)

    def finalize(self):
        """Compute aggregate statistics after all frames are processed."""
        if not self.frame_records:
            self.summary = {}
            return self.summary

        fps_list = [r["fps"] for r in self.frame_records if r["fps"] > 0]
        det_counts = [r["num_detections"] for r in self.frame_records]
        track_counts = [r["num_active_tracks"] for r in self.frame_records]

        # Track durations (in frames)
        durations = []
        for tid in self._track_first_frame:
            dur = self._track_last_frame[tid] - self._track_first_frame[tid] + 1
            durations.append(dur)

        self.summary = {
            "total_frames": len(self.frame_records),
            "total_unique_ids": len(self._all_ids),
            "total_id_switches": self._id_switch_count,
            "total_detections": int(np.sum(det_counts)) if det_counts else 0,
            "total_tracks": int(np.sum(track_counts)) if track_counts else 0,
            "avg_detections_per_frame": round(float(np.mean(det_counts)), 2),
            "max_detections_per_frame": int(np.max(det_counts)) if det_counts else 0,
            "avg_active_tracks": round(float(np.mean(track_counts)), 2),
            "avg_fps": round(float(np.mean(fps_list)), 2) if fps_list else 0.0,
            "median_fps": round(float(np.median(fps_list)), 2) if fps_list else 0.0,
            "min_fps": round(float(np.min(fps_list)), 2) if fps_list else 0.0,
            "max_fps": round(float(np.max(fps_list)), 2) if fps_list else 0.0,
            "avg_confidence": round(float(np.mean(self._all_scores)), 4) if self._all_scores else 0.0,
            "median_confidence": round(float(np.median(self._all_scores)), 4) if self._all_scores else 0.0,
            "avg_track_duration_frames": round(float(np.mean(durations)), 1) if durations else 0.0,
            "median_track_duration_frames": round(float(np.median(durations)), 1) if durations else 0.0,
            "max_track_duration_frames": int(np.max(durations)) if durations else 0,
            "min_track_duration_frames": int(np.min(durations)) if durations else 0,
        }
        return self.summary

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------
    def save_csv(self, path):
        """Save per-frame records to CSV."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if not self.frame_records:
            return
        keys = self.frame_records[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.frame_records)

    def save_json(self, path):
        """Save aggregate summary + per-frame records to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if not hasattr(self, "summary"):
            self.finalize()
        data = {
            "summary": self.summary,
            "per_frame": self.frame_records,
            "track_durations": {
                str(tid): self._track_last_frame[tid] - self._track_first_frame[tid] + 1
                for tid in self._track_first_frame
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self):
        """Pretty-print aggregate statistics to console."""
        if not hasattr(self, "summary") or not self.summary:
            self.finalize()
        print("\n" + "=" * 60)
        print("  TRACKING METRICS SUMMARY")
        print("=" * 60)
        for k, v in self.summary.items():
            label = k.replace("_", " ").title()
            print(f"  {label:.<45} {v}")
        print("=" * 60 + "\n")
