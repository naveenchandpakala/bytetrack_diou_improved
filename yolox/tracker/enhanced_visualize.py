import cv2
import numpy as np


def _get_color(idx):
    """Deterministic colour for a track ID – same palette as original."""
    idx = abs(int(idx)) * 3
    return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)


def plot_tracking_enhanced(
    image,
    tlwhs,
    obj_ids,
    scores=None,
    frame_id=0,
    fps=0.0,
    num_active_tracks=None,
):
    """
    Draw tracking results on *image* with enhanced HUD.

    Parameters
    ----------
    image : np.ndarray  (H, W, 3) BGR
    tlwhs : list of [x, y, w, h]
    obj_ids : list of int (track IDs)
    scores : list of float (confidence)  – may be None
    frame_id : int
    fps : float
    num_active_tracks : int or None (defaults to len(obj_ids))

    Returns
    -------
    im : np.ndarray  (same size, annotated)
    """
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    # Scale-adaptive sizes
    scale = max(im_w / 1920.0, 0.5)
    text_scale = max(scale * 1.2, 0.5)
    line_thickness = max(int(2 * scale), 1)
    text_thickness = max(int(1.5 * scale), 1)

    n_tracks = num_active_tracks if num_active_tracks is not None else len(obj_ids)
    n_dets = len(tlwhs)

    # -------------------------------------------------------------------
    # Overlay panel  (semi-transparent black rectangle in top-left)
    # -------------------------------------------------------------------
    panel_lines = [
        f"Frame: {frame_id}",
        f"FPS: {fps:.1f}",
        f"Active Tracks: {n_tracks}",
        f"Humans Detected: {n_dets}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = text_scale * 0.65
    pad = 8
    line_h = int(28 * scale)
    panel_h = pad + line_h * len(panel_lines) + pad
    panel_w = int(300 * scale)

    overlay = im.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, im, 0.45, 0, im)

    for idx, line in enumerate(panel_lines):
        y = pad + (idx + 1) * line_h - int(4 * scale)
        cv2.putText(im, line, (pad, y), font, font_scale,
                    (0, 255, 255), text_thickness, cv2.LINE_AA)

    # -------------------------------------------------------------------
    # Draw each tracked person
    # -------------------------------------------------------------------
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        x2, y2 = x1 + w, y1 + h
        intbox = (int(x1), int(y1), int(x2), int(y2))
        obj_id = int(obj_ids[i])
        color = _get_color(obj_id)

        # Bounding box
        cv2.rectangle(im, intbox[0:2], intbox[2:4],
                      color=color, thickness=line_thickness)

        # ID label above box
        id_text = f"ID: {obj_id}"
        id_size = cv2.getTextSize(id_text, font, font_scale, text_thickness)[0]
        id_y = max(intbox[1] - 6, id_size[1] + 2)
        # Background rectangle for readability
        cv2.rectangle(im,
                      (intbox[0], id_y - id_size[1] - 4),
                      (intbox[0] + id_size[0] + 4, id_y + 4),
                      color, -1)
        txt_color = (255, 255, 255)
        cv2.putText(im, id_text, (intbox[0] + 2, id_y),
                    font, font_scale, txt_color, text_thickness, cv2.LINE_AA)

        # Confidence label inside top of box
        if scores is not None and i < len(scores):
            conf_text = f"{scores[i]:.0%}"
            conf_size = cv2.getTextSize(conf_text, font, font_scale * 0.85, 1)[0]
            conf_y = intbox[1] + conf_size[1] + 6
            cv2.putText(im, conf_text,
                        (intbox[0] + 4, min(conf_y, intbox[3] - 2)),
                        font, font_scale * 0.85, (255, 255, 255), 1, cv2.LINE_AA)

    return im
