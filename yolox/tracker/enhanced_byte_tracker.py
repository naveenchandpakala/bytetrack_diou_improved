"""
Enhanced ByteTrack Tracker
============================
A backwards-compatible, improved version of the original BYTETracker that
integrates five modular improvements:

1. **Camera Motion Compensation (CMC)** – warps Kalman states with the
   inter-frame affine transform *before* prediction, so the Kalman filter
   only predicts object-level motion.
2. **ReID appearance embedding** – lightweight 512-dim embeddings are
   extracted per detection and smoothed per track (EMA).  The first-stage
   association fuses spatial + appearance cost.
3. **DIoU / CIoU matching** – replaces standard IoU with distance-IoU so
   that non-overlapping boxes still carry meaningful cost.
4. **Adaptive track buffer** – high-confidence, long-lived tracks survive
   longer in the lost pool, reducing fragmentation.
5. **Occlusion-aware track state** – a new ``Occluded`` state lets heavily-
   overlapping tracks coast through short occlusions without losing their ID.

All improvements are **off by default** and controlled via simple boolean
flags in ``EnhancedBYTETracker.__init__``, so you can ablate each one
independently.

Usage
-----
Swap the import in ``demo_track.py``::

    from yolox.tracker.enhanced_byte_tracker import EnhancedSTrack, EnhancedBYTETracker

Then instantiate::

    tracker = EnhancedBYTETracker(args, frame_rate=30)

The ``update()`` signature gains one new optional argument ``img`` (the raw
BGR frame), needed for CMC and ReID.  If not provided, those features are
silently skipped.
"""

import numpy as np
from collections import deque
import copy
import logging

from .kalman_filter import KalmanFilter
from .basetrack import BaseTrack, TrackState
try:
    from .camera_motion import CameraMotionCompensator
except ImportError:
    CameraMotionCompensator = None
from . import enhanced_matching as matching
from . import matching as base_matching

logger = logging.getLogger(__name__)


# ===================================================================
# Extended track states  (backward-compatible with original TrackState)
# ===================================================================
class ExtTrackState(TrackState):
    Occluded = 4       # NEW: temporarily occluded, keep predicting


# ===================================================================
# Enhanced STrack  – adds appearance features + adaptive buffer
# ===================================================================
class EnhancedSTrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None):
        # --- Original fields ------------------------------------------------
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0

        # --- Appearance features (Improvement 1: ReID) ----------------------
        self.smooth_feat = None      # EMA-smoothed feature
        self.curr_feat = None        # most recent feature
        self._feat_alpha = 0.9       # EMA decay (higher = more memory)
        if feat is not None:
            self.update_features(feat)

        # --- Adaptive buffer (Improvement 4) --------------------------------
        self.confidence_history = deque(maxlen=50)
        self.avg_score = score
        self.confidence_history.append(score)

        # --- Occlusion state (Improvement 5) --------------------------------
        self.occlusion_count = 0
        self.visibility = 1.0

    # ---------------------------------------------------------------
    # Appearance feature management
    # ---------------------------------------------------------------
    def update_features(self, feat):
        """Update appearance feature with EMA smoothing."""
        if feat is None:
            return
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = (self._feat_alpha * self.smooth_feat
                                + (1.0 - self._feat_alpha) * feat)
            self.smooth_feat /= (np.linalg.norm(self.smooth_feat) + 1e-8)

    # ---------------------------------------------------------------
    # Kalman prediction
    # ---------------------------------------------------------------
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) == 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = EnhancedSTrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.update_features(getattr(new_track, 'curr_feat', None))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.confidence_history.append(new_track.score)
        self.avg_score = np.mean(self.confidence_history)
        self.occlusion_count = 0
        self.visibility = 1.0

    def update(self, new_track, frame_id):
        """Update a matched track."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.update_features(getattr(new_track, 'curr_feat', None))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.confidence_history.append(new_track.score)
        self.avg_score = np.mean(self.confidence_history)
        self.occlusion_count = 0
        self.visibility = 1.0

    # ---------------------------------------------------------------
    # Occlusion-aware state  (Improvement 5)
    # ---------------------------------------------------------------
    def mark_occluded(self):
        """Transition to occluded state (keep predicting, don't change ID)."""
        self.state = ExtTrackState.Occluded
        self.occlusion_count += 1

    # ---------------------------------------------------------------
    # Adaptive track buffer  (Improvement 4)
    # ---------------------------------------------------------------
    @property
    def adaptive_max_lost(self):
        """
        How many frames this track can stay in the lost pool before removal.
        Higher confidence + longer age → more patience.
        """
        base = getattr(self, "base_buffer_size", 30)
        conf_bonus = int(self.avg_score * getattr(self, "confidence_buffer_scale", 20))
        age = self.frame_id - self.start_frame
        age_bonus = min(int(age / 30), getattr(self, "max_age_buffer_bonus", 20))
        return base + conf_bonus + age_bonus

    # ---------------------------------------------------------------
    # BBox property accessors  (identical to original STrack)
    # ---------------------------------------------------------------
    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame,
                                       self.end_frame)


# ===================================================================
# Enhanced BYTETracker
# ===================================================================
class EnhancedBYTETracker:
    """
    Improved BYTETracker with five modular improvements.

    Each improvement can be toggled independently for ablation:

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: track_thresh, track_buffer, match_thresh, mot20
    frame_rate : int
    enable_cmc : bool       Camera Motion Compensation
    enable_reid : bool      Appearance embedding in association
    enable_diou : bool      DIoU (instead of IoU) matching
    enable_adaptive_buf : bool  Adaptive lost-track buffer
    enable_occlusion : bool     Occlusion-aware track state
    reid_model_path : str or None   Path to ReID checkpoint
    lambda_iou : float      Weight of spatial cost in fused matching (0..1)
    occlusion_thresh : float  IoU overlap ratio above which a track is
                              considered occluded
    max_occlusion_frames : int  Max consecutive occluded frames before Lost
    """

    def __init__(
        self,
        args,
        frame_rate=30,
        # --- Feature toggles (all ON by default) ---
        enable_cmc=True,
        enable_reid=True,
        enable_diou=True,
        enable_adaptive_buf=True,
        enable_occlusion=True,
        # --- ReID config ---
        reid_model_path=None,
        # --- Matching config ---
        lambda_iou=0.5,
        confidence_lambda=0.7,
        min_dynamic_buffer=30,
        max_dynamic_buffer=90,
        confidence_buffer_scale=20,
        max_age_buffer_bonus=20,
        # --- Occlusion config ---
        occlusion_thresh=0.3,
        max_occlusion_frames=15,
    ):
        self.tracked_stracks = []   # type: list[EnhancedSTrack]
        self.lost_stracks = []      # type: list[EnhancedSTrack]
        self.removed_stracks = []   # type: list[EnhancedSTrack]

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # --- Feature toggles -----------------------------------------------
        self.enable_cmc = enable_cmc and CameraMotionCompensator is not None
        self.enable_reid = enable_reid
        self.enable_diou = enable_diou
        self.enable_adaptive_buf = enable_adaptive_buf
        self.enable_occlusion = enable_occlusion

        # --- Camera Motion Compensation  (Improvement 2) -------------------
        if enable_cmc and CameraMotionCompensator is None:
            logger.warning("camera_motion.py not found; disabling CMC.")
        self.cmc = CameraMotionCompensator() if self.enable_cmc else None

        # --- ReID extractor  (Improvement 1) --------------------------------
        self.reid_extractor = None
        if enable_reid:
            from .reid_extractor import ReIDExtractor
            self.reid_extractor = ReIDExtractor(
                model_path=reid_model_path, use_cuda=True)
            if not self.reid_extractor.is_available:
                logger.warning("ReID model unavailable — disabling appearance matching.")
                self.enable_reid = False

        self.lambda_iou = lambda_iou
        self.confidence_lambda = confidence_lambda
        self.min_dynamic_buffer = min_dynamic_buffer
        self.max_dynamic_buffer = max_dynamic_buffer
        self.confidence_buffer_scale = confidence_buffer_scale
        self.max_age_buffer_bonus = max_age_buffer_bonus
        self.occlusion_thresh = occlusion_thresh
        self.max_occlusion_frames = max_occlusion_frames

        # Log active features
        features = []
        if self.enable_cmc:    features.append("CMC")
        if self.enable_reid:   features.append("ReID")
        if self.enable_diou:   features.append("DIoU")
        if self.enable_adaptive_buf: features.append("AdaptiveBuf")
        if self.enable_occlusion:    features.append("Occlusion")
        logger.info("EnhancedBYTETracker active features: %s", features)

    # ===================================================================
    # Main update  (same pipeline as original, with enhancements injected)
    # ===================================================================
    def update(self, output_results, img_info, img_size, img=None):
        """
        Parameters
        ----------
        output_results : torch.Tensor or np.ndarray
            Detector output (N, 5) or (N, 7).
        img_info : list/tuple
            [height, width] of the original image.
        img_size : tuple
            Test-time input size (H, W).
        img : np.ndarray or None
            Raw BGR frame.  Needed for CMC and ReID. If None those features
            are silently skipped for this frame.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # ---------------------------------------------------------------
        # 0.  Parse detections
        # ---------------------------------------------------------------
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        # Split into high / low score detections (core ByteTrack idea)
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.05
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]

        # ---------------------------------------------------------------
        # 0b.  Extract ReID features for high-score detections
        # ---------------------------------------------------------------
        features_keep = None
        if self.enable_reid and img is not None and len(dets) > 0:
            features_keep = self.reid_extractor.extract(img, dets)

        # ---------------------------------------------------------------
        # 0c.  Create detection STrack objects
        # ---------------------------------------------------------------
        if len(dets) > 0:
            detections = []
            for idx, (tlbr, s) in enumerate(zip(dets, scores_keep)):
                feat = features_keep[idx] if features_keep is not None else None
                det = EnhancedSTrack(EnhancedSTrack.tlbr_to_tlwh(tlbr), s, feat=feat)
                det.base_buffer_size = self.min_dynamic_buffer
                det.confidence_buffer_scale = self.confidence_buffer_scale
                det.max_age_buffer_bonus = self.max_age_buffer_bonus
                detections.append(det)
        else:
            detections = []

        # ---------------------------------------------------------------
        # 1.  Separate confirmed / unconfirmed tracks
        # ---------------------------------------------------------------
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # ---------------------------------------------------------------
        # 2.  Pool tracked + lost tracks
        # ---------------------------------------------------------------
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # ---------------------------------------------------------------
        # 2b.  Camera Motion Compensation  (Improvement 2)
        # ---------------------------------------------------------------
        if self.enable_cmc and self.cmc is not None and img is not None:
            affine = self.cmc.compute_affine(img)
            CameraMotionCompensator.warp_stracks(strack_pool, affine)

        # ---------------------------------------------------------------
        # 2c.  Kalman prediction
        # ---------------------------------------------------------------
        EnhancedSTrack.multi_predict(strack_pool)

        # ---------------------------------------------------------------
        # 3.  First association  (high-score detections)
        # ---------------------------------------------------------------
        for track in strack_pool:
            track.base_buffer_size = self.min_dynamic_buffer
            track.confidence_buffer_scale = self.confidence_buffer_scale
            track.max_age_buffer_bonus = self.max_age_buffer_bonus

        if self.enable_reid and features_keep is not None and len(strack_pool) > 0 and len(detections) > 0:
            # Fused cost:  λ·spatial + (1-λ)·appearance
            emb_dists = matching.embedding_distance(strack_pool, detections)
        if self.enable_diou:
            dists = base_matching.diou_distance(strack_pool, detections)
        else:
            dists = base_matching.iou_distance(strack_pool, detections)

        if self.enable_reid and features_keep is not None and len(strack_pool) > 0 and len(detections) > 0:
            dists = self.lambda_iou * dists + (1.0 - self.lambda_iou) * emb_dists

        dists = self._apply_confidence_aware_cost(dists, detections)

        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)

        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # ---------------------------------------------------------------
        # 4.  Second association  (low-score detections, spatial only)
        # ---------------------------------------------------------------
        if len(dets_second) > 0:
            detections_second = [EnhancedSTrack(
                EnhancedSTrack.tlbr_to_tlwh(tlbr), s)
                for tlbr, s in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked = [strack_pool[i] for i in u_track
                     if strack_pool[i].state == TrackState.Tracked]

        if self.enable_diou:
            dists = base_matching.diou_distance(r_tracked, detections_second)
        else:
            dists = base_matching.iou_distance(r_tracked, detections_second)

        matches, u_track_second, u_det_second = matching.linear_assignment(
            dists, thresh=0.6)

        for itracked, idet in matches:
            track = r_tracked[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # ---------------------------------------------------------------
        # 4b.  Handle unmatched tracks → Lost / Occluded
        # ---------------------------------------------------------------
        for it in u_track_second:
            track = r_tracked[it]
            if track.state == TrackState.Lost:
                continue

            # --- Occlusion-aware state  (Improvement 5) ------------------
            if self.enable_occlusion:
                vis = self._estimate_visibility(track)
                track.visibility = vis
                if vis < self.occlusion_thresh:
                    # Heavily occluded — keep predicting, don't lose ID yet
                    track.mark_occluded()
                    if track.occlusion_count > self.max_occlusion_frames:
                        track.mark_lost()
                        lost_stracks.append(track)
                    # else: stays in tracked_stracks (will be predicted next frame)
                    continue

            track.mark_lost()
            lost_stracks.append(track)

        # ---------------------------------------------------------------
        # 5.  Deal with unconfirmed tracks
        # ---------------------------------------------------------------
        det_remaining = [detections[i] for i in u_detection]
        if self.enable_diou:
            dists = base_matching.diou_distance(unconfirmed, det_remaining)
        else:
            dists = base_matching.iou_distance(unconfirmed, det_remaining)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, det_remaining)

        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.8)

        for itracked, idet in matches:
            unconfirmed[itracked].update(det_remaining[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # ---------------------------------------------------------------
        # 6.  Init new stracks
        # ---------------------------------------------------------------
        for inew in u_detection:
            track = det_remaining[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # ---------------------------------------------------------------
        # 7.  Update state — remove stale lost tracks
        # ---------------------------------------------------------------
        for track in self.lost_stracks:
            if self.enable_adaptive_buf:
                max_lost = min(
                    self.max_dynamic_buffer,
                    max(self.min_dynamic_buffer, track.adaptive_max_lost),
                )
            else:
                max_lost = self.max_time_lost

            if self.frame_id - track.end_frame > max_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # ---------------------------------------------------------------
        # 8.  Bookkeeping  (identical to original)
        # ---------------------------------------------------------------
        self.tracked_stracks = [
            t for t in self.tracked_stracks
            if t.state == TrackState.Tracked or t.state == ExtTrackState.Occluded
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        output_stracks = [t for t in self.tracked_stracks if t.is_activated]
        return output_stracks

    def _apply_confidence_aware_cost(self, cost_matrix, detections):
        """Blend spatial cost with detection confidence."""
        if cost_matrix.size == 0 or len(detections) == 0:
            return cost_matrix

        det_scores = np.array([det.score for det in detections], dtype=np.float64)
        det_scores = np.clip(det_scores, 0.0, 1.0)
        confidence_cost = 1.0 - det_scores
        confidence_cost = np.expand_dims(confidence_cost, axis=0).repeat(
            cost_matrix.shape[0], axis=0)
        return (
            self.confidence_lambda * cost_matrix
            + (1.0 - self.confidence_lambda) * confidence_cost
        )

    # ---------------------------------------------------------------
    # Occlusion estimation helper
    # ---------------------------------------------------------------
    def _estimate_visibility(self, track):
        """
        Estimate visibility of *track* based on how much its bounding box
        overlaps with other tracked objects. Returns a float in [0, 1]
        where 1 = fully visible and 0 = fully occluded.
        """
        my_box = track.tlbr
        my_area = max((my_box[2] - my_box[0]) * (my_box[3] - my_box[1]), 1e-6)
        max_overlap = 0.0
        for other in self.tracked_stracks:
            if other.track_id == track.track_id:
                continue
            if other.state != TrackState.Tracked:
                continue
            ob = other.tlbr
            inter_x1 = max(my_box[0], ob[0])
            inter_y1 = max(my_box[1], ob[1])
            inter_x2 = min(my_box[2], ob[2])
            inter_y2 = min(my_box[3], ob[3])
            inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
            overlap = inter / my_area
            if overlap > max_overlap:
                max_overlap = overlap
        return 1.0 - max_overlap


# ===================================================================
# Utility functions  (identical to original byte_tracker.py)
# ===================================================================

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = base_matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
