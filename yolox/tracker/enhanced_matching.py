
import numpy as np
from scipy.spatial.distance import cdist

from yolox.tracker import matching as base_matching


# ---------------------------------------------------------------------------
# 1.  DIoU + shape-aware (CIoU-style aspect-ratio penalty) distance
# ---------------------------------------------------------------------------

def _diou_matrix(atlbrs, btlbrs, use_shape_penalty=True):
    """
    Compute the DIoU (or CIoU) between two sets of boxes (numpy, pure-Python).

    Parameters
    ----------
    atlbrs, btlbrs : np.ndarray
        (M, 4) and (N, 4) arrays of boxes in tlbr format.
    use_shape_penalty : bool
        If True, add CIoU-style aspect-ratio consistency penalty.

    Returns
    -------
    cost : np.ndarray   (M, N)  in [0, 1]  (0 = identical boxes)
    """
    M = len(atlbrs)
    N = len(btlbrs)
    cost = np.zeros((M, N), dtype=np.float64)
    if M == 0 or N == 0:
        return cost

    a = np.asarray(atlbrs, dtype=np.float64)
    b = np.asarray(btlbrs, dtype=np.float64)

    for i in range(M):
        ax1, ay1, ax2, ay2 = a[i]
        aw, ah = ax2 - ax1, ay2 - ay1
        a_cx, a_cy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0

        for j in range(N):
            bx1, by1, bx2, by2 = b[j]
            bw, bh = bx2 - bx1, by2 - by1
            b_cx, b_cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

            # --- IoU ---
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
            area_a = aw * ah
            area_b = bw * bh
            union = area_a + area_b - inter
            iou = inter / (union + 1e-7)

            # --- Center distance / enclosing diagonal ---
            d2 = (a_cx - b_cx) ** 2 + (a_cy - b_cy) ** 2
            enc_x1 = min(ax1, bx1)
            enc_y1 = min(ay1, by1)
            enc_x2 = max(ax2, bx2)
            enc_y2 = max(ay2, by2)
            c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

            diou = iou - d2 / c2

            # --- CIoU aspect-ratio penalty (optional) ---
            if use_shape_penalty:
                v = (4.0 / (np.pi ** 2)) * (
                    np.arctan(aw / (ah + 1e-7)) - np.arctan(bw / (bh + 1e-7))
                ) ** 2
                alpha = v / (1.0 - iou + v + 1e-7)
                diou -= alpha * v

            # Convert to cost ∈ [0, 1]:  diou ∈ [-1, 1]  →  cost ∈ [0, 1]
            cost[i, j] = (1.0 - diou) / 2.0

    return cost


def diou_distance(atracks, btracks, use_shape_penalty=True):
    """
    Drop-in replacement for ``matching.iou_distance`` that uses DIoU.

    Returns cost_matrix  (M, N) with values in [0, 1].
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    atlbrs = np.asarray(atlbrs, dtype=np.float64).reshape(-1, 4)
    btlbrs = np.asarray(btlbrs, dtype=np.float64).reshape(-1, 4)

    return _diou_matrix(atlbrs, btlbrs, use_shape_penalty=use_shape_penalty)


# ---------------------------------------------------------------------------
# 2.  Appearance (ReID) embedding distance
# ---------------------------------------------------------------------------

def embedding_distance(tracks, detections, metric='cosine'):
    """
    Compute appearance cost between tracks and detections.

    Tracks must have ``.smooth_feat`` and detections must have ``.curr_feat``.
    Both should be L2-normalized.

    Returns cost_matrix (M, N) in [0, 1] for cosine distance.
    """
    M = len(tracks)
    N = len(detections)
    cost_matrix = np.zeros((M, N), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([d.curr_feat for d in detections], dtype=np.float64)
    track_features = np.asarray([t.smooth_feat for t in tracks], dtype=np.float64)

    # Cosine distance ∈ [0, 2]; clip to [0, 1] for practical use
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    return cost_matrix


# ---------------------------------------------------------------------------
# 3.  Fused cost:  λ · IoU_cost  +  (1-λ) · appearance_cost
# ---------------------------------------------------------------------------

def fused_distance(atracks, detections, lambda_iou=0.5, use_diou=True):
    # --- Spatial cost ---
    if use_diou:
        iou_cost = diou_distance(atracks, detections)
    else:
        iou_cost = base_matching.iou_distance(atracks, detections)

    # --- Check if appearance features are available ---
    has_appearance = (
        len(atracks) > 0 and len(detections) > 0 and
        hasattr(atracks[0], 'smooth_feat') and atracks[0].smooth_feat is not None and
        hasattr(detections[0], 'curr_feat') and detections[0].curr_feat is not None
    )

    if not has_appearance:
        return iou_cost

    # --- Appearance cost ---
    emb_cost = embedding_distance(atracks, detections)

    # --- Adaptive per-track fusion ---
    cost_matrix = np.empty_like(iou_cost)
    for i, t in enumerate(atracks):
        if t.smooth_feat is not None:
            cost_matrix[i, :] = lambda_iou * iou_cost[i, :] + (1.0 - lambda_iou) * emb_cost[i, :]
        else:
            cost_matrix[i, :] = iou_cost[i, :]

    return cost_matrix


# ---------------------------------------------------------------------------
# 4.  Mahalanobis gating (unchanged logic, re-exported for convenience)
# ---------------------------------------------------------------------------

def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """Gate implausible matches using Mahalanobis distance."""
    return base_matching.gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position)


# ---------------------------------------------------------------------------
# 5.  Score-fused cost  (same as original, re-exported)
# ---------------------------------------------------------------------------

def fuse_score(cost_matrix, detections):
    """Multiply IoU similarity by detection confidence."""
    return base_matching.fuse_score(cost_matrix, detections)


# ---------------------------------------------------------------------------
# Re-export essentials so imports stay clean
# ---------------------------------------------------------------------------
linear_assignment = base_matching.linear_assignment
iou_distance = base_matching.iou_distance
ious = base_matching.ious
