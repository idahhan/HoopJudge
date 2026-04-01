"""ByteTrack wrapper for stable multi-person tracking.

Wraps ``supervision.ByteTrack`` to accept per-frame ``PersonDetection``
lists and annotate each detection with a stable integer ``track_id`` that
persists across frames even through brief occlusions.

Usage::

    from myogait.trackers import PersonByteTracker, TrackerConfig

    cfg     = TrackerConfig()
    tracker = PersonByteTracker(cfg, fps=60.0)
    for frame_bgr, persons in zip(frames, detections_per_frame):
        tracked = tracker.update(persons)   # List[PersonDetection] with track_id set
        # persons with no track (below threshold) get track_id=None

Why ByteTrack over IoU-only tracking
-------------------------------------
IoU-based trackers (like the existing ``_PlayerTracker`` in possession.py)
assign new IDs whenever a bbox jumps > threshold between consecutive frames —
which happens whenever a player is briefly occluded, the detector skips a
frame, or the player moves fast.  ByteTrack uses a Kalman filter to *predict*
where each track will be even when the detector misses it, and uses a two-pass
matching strategy (high-confidence detections first, then low-confidence) to
recover tracks through gaps.  This is exactly the problem causing P6→P25
relabels in the possession pipeline.

Implementation notes
--------------------
- ``supervision.ByteTrack`` is used as the core tracker (already in the venv).
- ``update()`` converts ``List[PersonDetection]`` → ``sv.Detections``,
  calls the ByteTracker, then matches the returned tracked bboxes back to the
  input ``PersonDetection`` objects by nearest-bbox IoU.
- Detections that survive tracking get their ``track_id`` set in-place.
  Detections that are filtered out (low confidence, new unconfirmed) keep
  ``track_id=None``.
- Track births, deaths, and ID reassignments are logged at DEBUG level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrackerConfig:
    """ByteTrack configuration parameters.

    All parameters mirror those exposed by ``supervision.ByteTrack``.

    Attributes
    ----------
    track_activation_threshold : float
        Minimum detection confidence to *activate* a new track.
        Detections below this are still used in the low-confidence matching
        pass to keep existing tracks alive.  Default: 0.25.
    lost_track_buffer : int
        How many frames to keep a *lost* track alive (using Kalman prediction)
        before discarding it.  Higher values tolerate longer occlusions but
        increase ID confusion risk.  Default: 30.
    minimum_matching_threshold : float
        IoU threshold for the high-confidence matching pass.  Default: 0.8.
    minimum_consecutive_frames : int
        A new detection must be matched for this many consecutive frames
        before its track ID is confirmed.  Default: 1 (no warmup).
    min_box_area : float
        Detections whose bbox area (pixels²) is below this are ignored.
        Default: 100.
    """

    track_activation_threshold:  float = 0.25
    lost_track_buffer:            int   = 30
    minimum_matching_threshold:   float = 0.80
    minimum_consecutive_frames:   int   = 1
    min_box_area:                 float = 100.0


# ---------------------------------------------------------------------------
# IoU helper (no external dependency)
# ---------------------------------------------------------------------------

def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of (x1, y1, x2, y2) boxes.

    Parameters
    ----------
    boxes_a : (N, 4)
    boxes_b : (M, 4)

    Returns
    -------
    iou : (N, M)
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Expand dims for broadcasting
    a = boxes_a[:, None, :]   # (N, 1, 4)
    b = boxes_b[None, :, :]   # (1, M, 4)

    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, ix2 - ix1)
    inter_h = np.maximum(0.0, iy2 - iy1)
    inter   = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class PersonByteTracker:
    """Stable multi-person tracker based on ByteTrack (via supervision).

    Parameters
    ----------
    config : TrackerConfig
        Tracker hyperparameters.
    fps : float
        Video frame rate.  Used to convert ``lost_track_buffer`` (frames)
        to seconds inside the ByteTracker.  Default: 30.0.
    """

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        fps: float = 30.0,
    ) -> None:
        self._cfg = config or TrackerConfig()
        self._fps = fps
        self._tracker = self._build_tracker()

        # Track bookkeeping for logging
        self._prev_track_ids: set = set()
        self._frame_idx: int = 0

    def _build_tracker(self):
        try:
            import supervision as sv
        except ImportError as e:
            raise ImportError(
                "PersonByteTracker requires supervision:\n  pip install supervision"
            ) from e

        return sv.ByteTrack(
            track_activation_threshold = self._cfg.track_activation_threshold,
            lost_track_buffer          = self._cfg.lost_track_buffer,
            minimum_matching_threshold = self._cfg.minimum_matching_threshold,
            frame_rate                 = int(self._fps),
            minimum_consecutive_frames = self._cfg.minimum_consecutive_frames,
        )

    def reset(self) -> None:
        """Reset tracker state (call between clips)."""
        self._tracker = self._build_tracker()
        self._prev_track_ids = set()
        self._frame_idx = 0

    def update(self, persons: "List") -> "List":
        """Assign stable track IDs to *persons* for the current frame.

        Parameters
        ----------
        persons : List[PersonDetection]
            Raw per-frame detections (from ``YOLOPersonDetector``).

        Returns
        -------
        List[PersonDetection]
            Same list with ``track_id`` set on matched detections.
            Detections that ByteTrack does not track (e.g. new unconfirmed
            tracks during warm-up) keep ``track_id = None``.
        """
        import supervision as sv

        fidx = self._frame_idx
        self._frame_idx += 1

        if not persons:
            self._log_losses(set(), fidx)
            return persons

        # Filter by min box area
        min_area = self._cfg.min_box_area
        valid    = [p for p in persons if p.area >= min_area]
        invalid  = [p for p in persons if p.area <  min_area]

        if not valid:
            self._log_losses(set(), fidx)
            return persons  # all too small — return as-is with track_id=None

        # Build supervision Detections
        xyxy       = np.array([p.bbox for p in valid], dtype=np.float32)
        confidence = np.array([p.confidence for p in valid], dtype=np.float32)
        class_ids  = np.zeros(len(valid), dtype=int)

        sv_dets = sv.Detections(
            xyxy       = xyxy,
            confidence = confidence,
            class_id   = class_ids,
        )

        tracked = self._tracker.update_with_detections(sv_dets)

        # Match tracked bboxes back to input PersonDetection objects via IoU
        cur_track_ids: set = set()
        if len(tracked) > 0 and tracked.tracker_id is not None:
            tracked_xyxy = tracked.xyxy            # (K, 4)
            tracked_tids = tracked.tracker_id      # (K,)
            input_xyxy   = xyxy                    # (N, 4)

            iou = _iou_matrix(input_xyxy, tracked_xyxy)  # (N, K)

            used_t = set()
            for n_idx in range(len(valid)):
                if iou.shape[1] == 0:
                    break
                t_idx = int(np.argmax(iou[n_idx]))
                if iou[n_idx, t_idx] > 0.30 and t_idx not in used_t:
                    valid[n_idx].track_id = int(tracked_tids[t_idx])
                    cur_track_ids.add(int(tracked_tids[t_idx]))
                    used_t.add(t_idx)

        self._log_changes(cur_track_ids, fidx)
        self._prev_track_ids = cur_track_ids

        # Persons that were filtered out (area < min) keep track_id=None
        return valid + invalid

    def _log_changes(self, cur_ids: set, fidx: int) -> None:
        born  = cur_ids - self._prev_track_ids
        lost  = self._prev_track_ids - cur_ids
        if born:
            logger.debug("f%04d  track BORN  %s", fidx, sorted(born))
        if lost:
            logger.debug("f%04d  track LOST  %s", fidx, sorted(lost))

    def _log_losses(self, cur_ids: set, fidx: int) -> None:
        lost = self._prev_track_ids - cur_ids
        if lost:
            logger.debug("f%04d  track LOST (no dets)  %s", fidx, sorted(lost))
        self._prev_track_ids = cur_ids
