"""Trajectory-based ball possession tracking.

Upgrades the frame-by-frame scoring in player_selection to a temporally-stable,
trajectory-aware possession tracker.

Key ideas
---------
1. **Player identity**: IoU-based cross-frame tracking gives each player a
   stable ID within the clip so trajectories accumulate consistently.

2. **Trajectory buffers**: Rolling windows (default 15 frames) of ball
   positions and each player's best wrist/hand positions.

3. **Velocity correlation**: The player whose wrist velocity direction matches
   the ball velocity direction most consistently over the window is the most
   likely ball handler.

4. **Distance stability**: A ball held or dribbled shows low variance in
   ball-to-wrist distance.  A free ball shows high variance or large distance.

5. **Hysteresis**: Once a player is assigned possession, a challenger must
   outscore them by a margin **and** maintain that lead for N consecutive
   frames before possession switches.  This prevents single-frame noise from
   causing spurious switches.

Scoring weights
---------------
  proximity   0.25  wrist distance to ball (immediate)
  trajectory  0.45  velocity-vector cosine similarity over window (temporal)
  stability   0.15  low variance in ball-to-wrist distance (temporal)
  hysteresis  0.15  was this the possessor last frame?

Entry points
------------
track_possession(data, config)
    Run possession tracking.  Requires data["player_selection"] and
    data["ball"] to exist (call analyze_ball + select_target_player first).
    Writes data["possession"] and returns data.

render_possession_video(video_path, data, output_path, ...)
    Debug video with possession overlay, trajectory trails, and score HUD.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = __import__("logging").getLogger(__name__)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    # Rolling window length in frames
    "trajectory_window": 15,
    # Scoring weights (must sum to 1.0)
    "w_proximity":   0.25,
    "w_trajectory":  0.45,
    "w_stability":   0.15,
    "w_hysteresis":  0.15,
    # Wrist search radius (pixels or multiples of ball radius)
    "wrist_search_radius_px":   120.0,
    "wrist_search_radius_ball": 7.0,
    # Possession switching rules
    "switch_threshold":   0.18,   # challenger must lead by at least this margin
    "min_switch_frames":  3,      # challenger must win for this many consecutive frames
    # Minimum ball speed (px/frame) to include a velocity sample
    "min_ball_speed":  0.5,
    # Minimum wrist speed (px/frame) to include a velocity sample
    "min_wrist_speed": 0.5,
    # IoU threshold to re-identify the same player across frames
    "player_iou_threshold": 0.30,
    # Tail length (frames) for trajectory trails in the debug video
    "trail_length": 12,
}


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def _iou(a: Tuple, b: Tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / max(union, 1e-6)


# ---------------------------------------------------------------------------
# IoU-based player identity tracker
# ---------------------------------------------------------------------------

class _PlayerTracker:
    """Assigns stable integer IDs to bounding boxes across frames via IoU."""

    def __init__(self, iou_threshold: float = 0.30) -> None:
        self.iou_threshold = iou_threshold
        self._next_id      = 0
        self._prev: Dict[int, Tuple] = {}   # id → bbox

    def update(self, bboxes: List[Tuple]) -> List[int]:
        """Return a stable ID for each bbox in *bboxes*."""
        ids: List[Optional[int]] = [None] * len(bboxes)

        if self._prev:
            candidates = []
            for i, bb in enumerate(bboxes):
                for pid, pb in self._prev.items():
                    score = _iou(bb, pb)
                    if score >= self.iou_threshold:
                        candidates.append((score, i, pid))
            candidates.sort(reverse=True)

            used_bbox = set()
            used_pid  = set()
            for _, i, pid in candidates:
                if i not in used_bbox and pid not in used_pid:
                    ids[i] = pid
                    used_bbox.add(i)
                    used_pid.add(pid)

        for i in range(len(bboxes)):
            if ids[i] is None:
                ids[i] = self._next_id
                self._next_id += 1

        self._prev = {ids[i]: bboxes[i] for i in range(len(bboxes))}
        return ids  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Trajectory buffer
# ---------------------------------------------------------------------------

class _TrajectoryBuffer:
    """Rolling trajectory buffers (pixel space) for ball and each player."""

    def __init__(self, window: int = 15) -> None:
        self.window  = window
        self.ball:    deque = deque(maxlen=window)
        self.players: Dict[int, deque] = {}

    def push_ball(self, pos: Optional[Tuple[float, float]]) -> None:
        self.ball.append(pos)

    def push_player(self, pid: int, pos: Optional[Tuple[float, float]]) -> None:
        if pid not in self.players:
            self.players[pid] = deque(maxlen=self.window)
        self.players[pid].append(pos)

    # ------------------------------------------------------------------

    def velocity_correlation(
        self,
        pid: int,
        min_valid: int = 3,
        min_ball_spd: float = 0.5,
        min_wrist_spd: float = 0.5,
    ) -> float:
        """Mean cosine similarity between ball velocity and wrist velocity.

        Returns value in [0, 1] where 0.5 = uncorrelated, 1.0 = perfectly
        aligned, 0.0 = anti-aligned.
        """
        ball_pos   = list(self.ball)
        wrist_pos  = list(self.players.get(pid, deque()))
        n = min(len(ball_pos), len(wrist_pos))
        if n < min_valid + 1:
            return 0.5

        sims: List[float] = []
        for t in range(1, n):
            bp  = ball_pos[t]
            bp0 = ball_pos[t - 1]
            wp  = wrist_pos[t]
            wp0 = wrist_pos[t - 1]
            if bp is None or bp0 is None or wp is None or wp0 is None:
                continue
            bv = (bp[0]-bp0[0], bp[1]-bp0[1])
            wv = (wp[0]-wp0[0], wp[1]-wp0[1])
            mb = math.sqrt(bv[0]**2 + bv[1]**2)
            mw = math.sqrt(wv[0]**2 + wv[1]**2)
            if mb < min_ball_spd or mw < min_wrist_spd:
                continue
            cos = max(-1.0, min(1.0, (bv[0]*wv[0] + bv[1]*wv[1]) / (mb * mw)))
            sims.append(cos)

        if len(sims) < min_valid:
            return 0.5
        return (sum(sims) / len(sims) + 1.0) / 2.0

    def distance_stability(self, pid: int) -> float:
        """Low variance in ball-to-wrist distance → likely possession.

        Returns [0, 1] where 1 = perfectly stable distance (strong signal).
        """
        ball_pos  = list(self.ball)
        wrist_pos = list(self.players.get(pid, deque()))
        n = min(len(ball_pos), len(wrist_pos))
        dists: List[float] = []
        for t in range(n):
            if ball_pos[t] is not None and wrist_pos[t] is not None:
                d = math.sqrt(
                    (ball_pos[t][0] - wrist_pos[t][0])**2 +
                    (ball_pos[t][1] - wrist_pos[t][1])**2
                )
                dists.append(d)

        if len(dists) < 3:
            return 0.5

        mean_d = sum(dists) / len(dists)
        std_d  = math.sqrt(sum((d - mean_d)**2 for d in dists) / len(dists))

        # Low std AND low mean → high stability score
        stability  = max(0.0, 1.0 - std_d  / 150.0)
        closeness  = max(0.0, 1.0 - mean_d / 300.0)
        return min(1.0, (stability + closeness) / 2.0)


# ---------------------------------------------------------------------------
# Main possession tracker
# ---------------------------------------------------------------------------

class PossessionTracker:
    """Stateful trajectory-based ball possession tracker.

    Maintains rolling trajectory buffers for the ball and each player's
    best wrist.  Uses velocity correlation and distance stability as the
    primary possession signals, with hysteresis to prevent spurious switches.

    Parameters
    ----------
    config : dict, optional
        Override any key from ``_DEFAULT_CONFIG``.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self._cfg = {**_DEFAULT_CONFIG, **(config or {})}
        self._traj   = _TrajectoryBuffer(self._cfg["trajectory_window"])
        self._id_tracker = _PlayerTracker(self._cfg["player_iou_threshold"])

        # Possession state
        self._possessor_id:  Optional[int] = None
        self._confidence:    float = 0.0
        self._switch_cand:   Optional[int] = None
        self._switch_frames: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        players: List[Dict],
        ball:    Dict,
        frame_idx: int,
        frame_w: int,
        frame_h: int,
    ) -> Dict:
        """Process one frame and return a possession dict.

        Parameters
        ----------
        players : list of player dicts
            From ``data["player_selection"]["per_frame"][i]["players"]``.
        ball : dict
            The ``"ball"`` entry from ``data["ball"]["per_frame"][i]["ball"]``.
        frame_idx : int
        frame_w, frame_h : int
            Frame pixel dimensions.

        Returns
        -------
        dict
            See module docstring for output schema.
        """
        cfg = self._cfg

        # --- Update ball trajectory ---
        bc = ball.get("center")
        ball_tracked = ball.get("tracked", False)
        self._traj.push_ball(tuple(bc) if bc and ball_tracked else None)

        # --- Player identity + trajectory update ---
        # Use ByteTrack track IDs when available (set by select_target_player);
        # fall back to IoU-based tracking for players without a stable track ID.
        bboxes = [tuple(p["bbox"]) for p in players if p.get("bbox")]
        if bboxes and all(p.get("track_id") is not None for p in players if p.get("bbox")):
            # All players have ByteTrack IDs — use them directly
            pids = [p["track_id"] for p in players if p.get("bbox")]
        else:
            # Fallback: IoU-based ID assignment
            pids = self._id_tracker.update(bboxes)

        for pid, p in zip(pids, players):
            wrist_px = self._best_wrist_px(p, bc, frame_w, frame_h)
            self._traj.push_player(pid, wrist_px)

        # --- No ball or no players → hold current ---
        if not players or not ball_tracked or bc is None:
            return self._hold_result(frame_idx, players, pids if players else [],
                                     "no ball tracked")

        # --- Score each player ---
        scored = []
        for pid, p in zip(pids, players):
            s = self._score(pid, p, ball, frame_w, frame_h)
            scored.append((pid, p, s))

        return self._update_possession(scored, pids, players, frame_idx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _best_wrist_px(
        self,
        player: Dict,
        ball_center: Optional[List],
        fw: int, fh: int,
    ) -> Optional[Tuple[float, float]]:
        """Return pixel coords of the wrist closest to the ball."""
        lw = player.get("left_wrist_norm")
        rw = player.get("right_wrist_norm")
        wrists = [(lw, "L"), (rw, "R")]
        valid  = [(w, s) for w, s in wrists if w is not None]
        if not valid:
            return None
        if len(valid) == 1 or ball_center is None:
            w = valid[0][0]
            return (w[0] * fw, w[1] * fh)
        # Pick wrist closer to ball
        def dist(wn):
            return math.sqrt((wn[0]*fw - ball_center[0])**2 +
                             (wn[1]*fh - ball_center[1])**2)
        best = min(valid, key=lambda x: dist(x[0]))[0]
        return (best[0] * fw, best[1] * fh)

    def _score(
        self,
        pid: int,
        player: Dict,
        ball: Dict,
        fw: int, fh: int,
    ) -> Dict[str, float]:
        cfg = self._cfg
        bc  = ball.get("center")
        br  = ball.get("radius") or 20.0

        # 1. Proximity
        prox = 0.0
        if bc is not None:
            sr = max(br * cfg["wrist_search_radius_ball"], cfg["wrist_search_radius_px"])
            wp = self._best_wrist_px(player, bc, fw, fh)
            if wp is not None:
                d     = math.sqrt((wp[0] - bc[0])**2 + (wp[1] - bc[1])**2)
                prox  = max(0.0, 1.0 - d / sr)

        # 2. Trajectory velocity correlation
        traj = self._traj.velocity_correlation(
            pid,
            min_ball_spd=cfg["min_ball_speed"],
            min_wrist_spd=cfg["min_wrist_speed"],
        )

        # 3. Distance stability
        stab = self._traj.distance_stability(pid)

        # 4. Hysteresis
        hyst = 1.0 if pid == self._possessor_id else 0.0

        total = min(1.0, max(0.0,
            cfg["w_proximity"]  * prox +
            cfg["w_trajectory"] * traj +
            cfg["w_stability"]  * stab +
            cfg["w_hysteresis"] * hyst
        ))

        return {
            "total":      total,
            "proximity":  prox,
            "trajectory": traj,
            "stability":  stab,
            "hysteresis": hyst,
        }

    def _update_possession(
        self,
        scored: List[Tuple[int, Dict, Dict]],
        pids: List[int],
        players: List[Dict],
        frame_idx: int,
    ) -> Dict:
        cfg = self._cfg
        scored_sorted = sorted(scored, key=lambda x: x[2]["total"], reverse=True)
        top_pid, top_player, top_s = scored_sorted[0]
        second_total = scored_sorted[1][2]["total"] if len(scored_sorted) > 1 else 0.0
        margin = top_s["total"] - second_total

        # --- First frame ---
        if self._possessor_id is None:
            self._possessor_id  = top_pid
            self._confidence    = top_s["total"]
            reason = (f"initial: prox={top_s['proximity']:.2f} "
                      f"traj={top_s['trajectory']:.2f}")
            return self._build_result(frame_idx, top_pid, top_player, top_s,
                                      scored, pids, players, reason)

        # --- Same player leading ---
        if top_pid == self._possessor_id:
            self._confidence    = top_s["total"]
            self._switch_cand   = None
            self._switch_frames = 0
            reason = (f"holding: prox={top_s['proximity']:.2f} "
                      f"traj={top_s['trajectory']:.2f} "
                      f"stab={top_s['stability']:.2f}")
            return self._build_result(frame_idx, top_pid, top_player, top_s,
                                      scored, pids, players, reason)

        # --- Different player leading ---
        if top_pid == self._switch_cand:
            self._switch_frames += 1
        else:
            self._switch_cand   = top_pid
            self._switch_frames = 1

        if (self._switch_frames >= cfg["min_switch_frames"] and
                margin >= cfg["switch_threshold"]):
            # Commit the switch
            old = self._possessor_id
            self._possessor_id  = top_pid
            self._confidence    = top_s["total"]
            self._switch_cand   = None
            self._switch_frames = 0
            reason = (f"SWITCH P{old}→P{top_pid}: "
                      f"margin={margin:.2f} over {cfg['min_switch_frames']} frames "
                      f"traj={top_s['trajectory']:.2f}")
            return self._build_result(frame_idx, top_pid, top_player, top_s,
                                      scored, pids, players, reason)

        # --- Hold: not enough evidence to switch yet ---
        curr_entry = next(((p, pl, s) for p, pl, s in scored
                           if p == self._possessor_id), None)
        if curr_entry is not None:
            curr_pid, curr_player, curr_s = curr_entry
            reason = (f"holding P{self._possessor_id}: "
                      f"challenger P{top_pid} for "
                      f"{self._switch_frames}/{cfg['min_switch_frames']} frames "
                      f"margin={margin:.2f}")
            return self._build_result(frame_idx, curr_pid, curr_player, curr_s,
                                      scored, pids, players, reason)
        else:
            # Current possessor not detected this frame
            reason = (f"possessor P{self._possessor_id} not visible, "
                      f"holding ({self._switch_frames}/{cfg['min_switch_frames']})")
            return self._hold_result(frame_idx, players, pids, reason)

    def _build_result(
        self,
        frame_idx: int,
        pid: int,
        player: Dict,
        scores: Dict,
        all_scored: List,
        pids: List[int],
        players: List[Dict],
        reason: str,
    ) -> Dict:
        idx_in_frame = pids.index(pid) if pid in pids else None
        return {
            "frame_idx":          frame_idx,
            "possessor_player_id": pid,
            "possessor_idx":      idx_in_frame,
            "possessor_bbox":     list(player["bbox"]) if player.get("bbox") else None,
            "confidence":         round(scores["total"], 4),
            "reason":             reason,
            "scores": [
                {
                    "player_id": p,
                    "bbox":       list(pl["bbox"]) if pl.get("bbox") else None,
                    "total":      round(s["total"],      4),
                    "proximity":  round(s["proximity"],  4),
                    "trajectory": round(s["trajectory"], 4),
                    "stability":  round(s["stability"],  4),
                }
                for p, pl, s in all_scored
            ],
        }

    def _hold_result(
        self,
        frame_idx: int,
        players: List[Dict],
        pids: List[int],
        reason: str,
    ) -> Dict:
        # Try to find the current possessor in this frame
        idx_in_frame = None
        possessor_bbox = None
        for i, pid in enumerate(pids):
            if pid == self._possessor_id:
                idx_in_frame   = i
                possessor_bbox = list(players[i]["bbox"]) if players[i].get("bbox") else None
                break
        return {
            "frame_idx":           frame_idx,
            "possessor_player_id": self._possessor_id,
            "possessor_idx":       idx_in_frame,
            "possessor_bbox":      possessor_bbox,
            "confidence":          round(self._confidence * 0.85, 4),
            "reason":              reason,
            "scores":              [],
        }


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def track_possession(data: dict, config: Optional[Dict] = None) -> dict:
    """Run trajectory-based possession tracking on pre-selected player data.

    Requires
    --------
    - ``data["player_selection"]`` — from :func:`select_target_player`.
    - ``data["ball"]``             — from :func:`analyze_ball`.

    Writes
    ------
    ``data["possession"]`` with keys ``config``, ``summary``, ``per_frame``.

    Parameters
    ----------
    data : dict
        Pivot data dict.
    config : dict, optional
        Override any key from ``_DEFAULT_CONFIG``.

    Returns
    -------
    dict  — same *data* with ``data["possession"]`` added.
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}

    ps_frames = data.get("player_selection", {}).get("per_frame", [])
    if not ps_frames:
        raise ValueError(
            "data['player_selection'] is empty. "
            "Run select_target_player() before track_possession()."
        )

    ball_map = {
        f["frame_idx"]: f.get("ball", {})
        for f in data.get("ball", {}).get("per_frame", [])
    }
    meta    = data.get("meta", {})
    frame_w = meta.get("width",  1920)
    frame_h = meta.get("height", 1080)
    fps     = meta.get("fps", 30.0)

    tracker = PossessionTracker(config=cfg)
    per_frame_out: List[Dict] = []

    for pf in ps_frames:
        frame_idx   = pf["frame_idx"]
        players     = pf.get("players", [])
        ball        = ball_map.get(frame_idx, {})
        result      = tracker.update(players, ball, frame_idx, frame_w, frame_h)
        per_frame_out.append(result)

    n_total     = len(per_frame_out)
    n_with_poss = sum(1 for r in per_frame_out if r.get("possessor_bbox"))
    n_switches  = sum(1 for r in per_frame_out if r.get("reason", "").startswith("SWITCH"))

    # Possessor time breakdown
    possessor_counts: Dict[Optional[int], int] = {}
    for r in per_frame_out:
        pid = r.get("possessor_player_id")
        possessor_counts[pid] = possessor_counts.get(pid, 0) + 1

    data["possession"] = {
        "config":  {k: v for k, v in cfg.items() if not callable(v)},
        "summary": {
            "n_frames":               n_total,
            "n_frames_with_possession": n_with_poss,
            "n_possession_switches":  n_switches,
            "possession_coverage_rate": round(n_with_poss / max(n_total, 1), 4),
            "fps":                    fps,
            "possessor_frame_counts": {
                str(k): v for k, v in possessor_counts.items()
            },
        },
        "per_frame": per_frame_out,
    }

    logger.info(
        "Possession tracking done: coverage=%.1f%% switches=%d",
        100 * n_with_poss / max(n_total, 1), n_switches,
    )
    return data


# ---------------------------------------------------------------------------
# Debug video renderer
# ---------------------------------------------------------------------------

_CLR_ALL         = (80,  80,  80)    # grey     — non-possessors
_CLR_POSSESSOR   = (0,  215, 255)    # gold     — ball possessor
_CLR_BALL        = (0,  140, 255)    # orange   — ball
_CLR_BALL_INTERP = (0,  255, 255)    # yellow   — interpolated ball
_CLR_TRAIL_BALL  = (0,  120, 200)    # dim orange — ball trail
_CLR_TRAIL_WRIST = (180, 50, 200)    # purple   — wrist trail
_CLR_HUD         = (230, 230, 230)
_CLR_LABEL_BG    = (20,  20,  20)
_CLR_HIGH_CONF   = (0,  220,   0)    # green    — high confidence
_CLR_MED_CONF    = (0,  200, 255)    # yellow
_CLR_LOW_CONF    = (0,   80, 220)    # red


def _conf_color(conf: float) -> tuple:
    if conf >= 0.65:
        return _CLR_HIGH_CONF
    if conf >= 0.45:
        return _CLR_MED_CONF
    return _CLR_LOW_CONF


def render_possession_video(
    video_path: str,
    data: dict,
    output_path: str,
    fps: Optional[float] = None,
    codec: str = "mp4v",
    show_trails: bool = True,
    show_scores: bool = True,
) -> str:
    """Render a possession-tracking debug video.

    Overlay
    -------
    - Thin grey box     — non-possessing players
    - Thick gold box    — ball possessor (colour-coded by confidence)
    - Ball circle       — orange=detected, yellow=interpolated
    - Wrist trail       — purple fading dots (possessor's wrist history)
    - Ball trail        — dim orange fading dots (ball history)
    - HUD               — frame, player ID, confidence, reason snippet
    - Score table       — per-player trajectory/proximity/stability scores

    Parameters
    ----------
    video_path : str
    data : dict  — needs data["possession"] and data["ball"]
    output_path : str
    fps : float, optional
    codec : str
    show_trails : bool
    show_scores : bool

    Returns
    -------
    str — path to written video.
    """
    poss    = data.get("possession", {})
    pf_map  = {r["frame_idx"]: r for r in poss.get("per_frame", [])}
    ball_map = {
        f["frame_idx"]: f.get("ball", {})
        for f in data.get("ball", {}).get("per_frame", [])
    }
    trail_len = poss.get("config", {}).get("trail_length",
                _DEFAULT_CONFIG["trail_length"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps, (W, H),
    )

    # Rolling trail buffers
    ball_trail:  deque = deque(maxlen=trail_len)
    wrist_trail: deque = deque(maxlen=trail_len)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pr   = pf_map.get(frame_idx, {})
        ball = ball_map.get(frame_idx, {})

        bc = ball.get("center")
        ball_tracked = ball.get("tracked", False)
        ball_src = ball.get("source", "detected")

        poss_bbox  = pr.get("possessor_bbox")
        poss_pid   = pr.get("possessor_player_id")
        confidence = pr.get("confidence", 0.0)
        reason     = pr.get("reason", "")
        all_scores = pr.get("scores", [])

        # Update trails
        ball_trail.append(tuple(bc) if bc and ball_tracked else None)

        # Find best wrist of possessor from scores
        poss_wrist_norm = None
        if poss_pid is not None:
            players_in_frame = data.get("player_selection", {}).get("per_frame", [])
            ps_entry = next(
                (p for p in players_in_frame if p.get("frame_idx") == frame_idx), {}
            )
            for p in ps_entry.get("players", []):
                pb = p.get("bbox")
                if pb and poss_bbox and _iou(tuple(pb), tuple(poss_bbox)) > 0.5:
                    lw = p.get("left_wrist_norm")
                    rw = p.get("right_wrist_norm")
                    if lw and rw and bc:
                        dl = math.sqrt((lw[0]*W-bc[0])**2 + (lw[1]*H-bc[1])**2)
                        dr = math.sqrt((rw[0]*W-bc[0])**2 + (rw[1]*H-bc[1])**2)
                        poss_wrist_norm = lw if dl < dr else rw
                    else:
                        poss_wrist_norm = lw or rw
                    break

        if poss_wrist_norm:
            wrist_trail.append((int(poss_wrist_norm[0]*W), int(poss_wrist_norm[1]*H)))
        else:
            wrist_trail.append(None)

        # --- Draw ball trail ---
        if show_trails:
            trail_list = list(ball_trail)
            for ti, tp in enumerate(trail_list):
                if tp is None:
                    continue
                alpha = (ti + 1) / len(trail_list)
                r = max(3, int(6 * alpha))
                col = tuple(int(c * alpha) for c in _CLR_TRAIL_BALL)
                cv2.circle(frame, (int(tp[0]), int(tp[1])), r, col, -1)

        # --- Draw wrist trail ---
        if show_trails:
            trail_list = list(wrist_trail)
            for ti, tp in enumerate(trail_list):
                if tp is None:
                    continue
                alpha = (ti + 1) / len(trail_list)
                r = max(2, int(5 * alpha))
                col = tuple(int(c * alpha) for c in _CLR_TRAIL_WRIST)
                cv2.circle(frame, tp, r, col, -1)

        # --- Non-possessor players ---
        for s in all_scores:
            sb = s.get("bbox")
            if sb is None:
                continue
            if poss_bbox and _iou(tuple(sb), tuple(poss_bbox)) > 0.5:
                continue
            cv2.rectangle(frame, (sb[0], sb[1]), (sb[2], sb[3]), _CLR_ALL, 1)

        # --- Possessor ---
        if poss_bbox:
            col = _conf_color(confidence)
            cv2.rectangle(frame,
                          (poss_bbox[0], poss_bbox[1]),
                          (poss_bbox[2], poss_bbox[3]),
                          col, 3)
            pid_label = f"P{poss_pid}"
            cv2.putText(frame, pid_label,
                        (poss_bbox[0], max(poss_bbox[1]-6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        # --- Ball ---
        if bc and ball_tracked:
            ball_col = _CLR_BALL_INTERP if ball_src == "interpolated" else _CLR_BALL
            br = max(int(ball.get("radius", 15)), 8)
            cv2.circle(frame, (int(bc[0]), int(bc[1])), br, ball_col, 2)
            cv2.circle(frame, (int(bc[0]), int(bc[1])), 3,  ball_col, -1)

        # --- Score table ---
        if show_scores and all_scores:
            y0 = H - 14 - 16 * len(all_scores)
            for si, s in enumerate(sorted(all_scores, key=lambda x: -x["total"])):
                line = (f"P{s['player_id']} "
                        f"tot={s['total']:.2f} "
                        f"prx={s['proximity']:.2f} "
                        f"trj={s['trajectory']:.2f} "
                        f"stb={s['stability']:.2f}")
                y = y0 + si * 16
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cv2.rectangle(frame, (6, y-th-2), (6+tw+4, y+3), _CLR_LABEL_BG, -1)
                cv2.putText(frame, line, (8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, _CLR_HUD, 1, cv2.LINE_AA)

        # --- HUD ---
        reason_short = reason[:60] + ("…" if len(reason) > 60 else "")
        hud_lines = [
            f"f{frame_idx:04d}  P{poss_pid}  conf={confidence:.2f}",
            reason_short,
        ]
        for li, line in enumerate(hud_lines):
            cv2.putText(frame, line, (10, 22 + li * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, _CLR_HUD, 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info("Possession video written → %s (%d frames)", output_path, frame_idx)
    return output_path
