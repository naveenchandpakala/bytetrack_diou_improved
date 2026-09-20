"""
Lightweight ReID Feature Extractor
====================================
Re-uses the small ResNet-style network already present in the ByteTrack
codebase (``yolox/deepsort_tracker/reid_model.py``).  That model outputs a
512-dim L2-normalized embedding from a 128×64 crop.

If no pretrained weight file is provided, the module falls back to a
*feature-free* mode and the tracker gracefully ignores appearance cues.

The extractor is intentionally kept **separate** from the tracker so it can
be swapped for any other backbone (e.g. OSNet, BoT, ResNet-50-IBN) without
touching tracking logic.

Typical latency: ~2-4 ms per frame (batch of ~30 crops on a single GPU).
"""

import os
import logging
import numpy as np
import cv2
import torch
import torchvision.transforms as T

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Try to import the lightweight Net already bundled with ByteTrack
# ---------------------------------------------------------------------------
try:
    from yolox.deepsort_tracker.reid_model import Net as DeepSORTNet
    _HAS_DEEPSORT_NET = True
except ImportError:
    _HAS_DEEPSORT_NET = False


class ReIDExtractor:
    """
    Extract fixed-length L2-normalized appearance embeddings for bbox crops.

    Parameters
    ----------
    model_path : str or None
        Path to a ``.pth`` or ``.pt`` checkpoint.
        - For the bundled DeepSORT net the checkpoint is expected to contain
          ``state_dict['net_dict']``.
        - If *None* or the file does not exist, the extractor runs in
          **dummy mode**: ``extract()`` returns None, and the tracker
          falls back to spatial-only matching.
    use_cuda : bool
    """

    def __init__(self, model_path=None, use_cuda=True):
        self.device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'
        self.model = None
        self.feat_dim = 512               # output dimension of DeepSORTNet
        self.input_size = (64, 128)       # (W, H) — Market-1501 convention

        self.norm = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # --- Try to load the model ----------------------------------------
        if model_path is not None and os.path.isfile(model_path):
            self._load_model(model_path)
        else:
            logger.warning(
                "ReID model path not provided or not found (%s). "
                "Running in feature-free mode (spatial-only matching).",
                model_path
            )

    def _load_model(self, model_path):
        if not _HAS_DEEPSORT_NET:
            logger.warning("Cannot import DeepSORT Net architecture. "
                           "Running in feature-free mode.")
            return

        try:
            self.model = DeepSORTNet(reid=True)
            ckpt = torch.load(model_path, map_location=self.device)
            # The bundled checkpoint wraps weights in 'net_dict'
            if isinstance(ckpt, dict) and 'net_dict' in ckpt:
                self.model.load_state_dict(ckpt['net_dict'])
            elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                self.model.load_state_dict(ckpt['state_dict'])
            else:
                self.model.load_state_dict(ckpt)
            self.model.to(self.device).eval()
            logger.info("Loaded ReID model from %s (device=%s)", model_path, self.device)
        except Exception as e:
            logger.warning("Failed to load ReID model: %s. Feature-free mode.", e)
            self.model = None

    @property
    def is_available(self):
        """True if a model was loaded successfully."""
        return self.model is not None

    def _preprocess_crop(self, crop):
        """Resize a single BGR crop to input_size and normalize."""
        resized = cv2.resize(crop.astype(np.float32) / 255.0,
                             self.input_size,                     # (W, H)
                             interpolation=cv2.INTER_LINEAR)
        # cv2 gives (H, W, C) BGR — convert to RGB tensor
        resized = resized[:, :, ::-1].copy()   # BGR → RGB
        tensor = self.norm(resized)
        return tensor

    @torch.no_grad()
    def extract(self, img, tlbrs):
        """
        Extract appearance features for a list of bounding boxes.

        Parameters
        ----------
        img : np.ndarray
            Full BGR frame (H, W, 3).
        tlbrs : list[np.ndarray] or np.ndarray
            (N, 4) bounding boxes in (x1, y1, x2, y2) format.

        Returns
        -------
        features : np.ndarray (N, feat_dim) or None
            L2-normalized feature vectors.  None if the model is unavailable
            or there are no valid boxes.
        """
        if self.model is None:
            return None

        if len(tlbrs) == 0:
            return None

        h, w = img.shape[:2]
        crops = []
        for box in tlbrs:
            x1, y1, x2, y2 = map(int, box)
            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                # Use a tiny black crop as placeholder
                crop = np.zeros((self.input_size[1], self.input_size[0], 3),
                                dtype=np.uint8)
            crops.append(self._preprocess_crop(crop))

        batch = torch.stack(crops, dim=0).to(self.device)
        features = self.model(batch).cpu().numpy()

        # L2-normalize (the model already does this in reid=True mode,
        # but an extra normalisation is a safety net)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-8)
        return features
