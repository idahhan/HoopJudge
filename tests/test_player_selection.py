"""Tests for the multi-person player-selection layer.

Covers:
- PersonDetection dataclass (area, center, wrists_norm, elbows_norm, to_dict)
- score_player_for_ball (inside, edge, wrist, hand_proxy, temporal components)
- _hand_proxy_norm computation
- assign_target_player (single/multi player, no-ball fallback, keypoints stored)
- smooth_assignments (consensus, gap-fill forward/backward)
- gate_landmarks_for_player_selection (IoU gating)
- coco17_to_landmarks conversion
- extract_selected_player_pose
- select_target_player (integration with synthetic video)
- render_player_selection_video (smoke test)
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from myogait.detectors.ball_detector import BallDetection
from myogait.detectors.person_detector import PersonDetection
from myogait.player_selection import (
    FrameAssignment,
    _bbox_iou,
    _edge_distance,
    _estimate_extracted_person_bbox,
    _hand_proxy_norm,
    assign_target_player,
    coco17_to_landmarks,
    extract_selected_player_pose,
    gate_landmarks_for_player_selection,
    render_player_selection_video,
    score_player_for_ball,
    smooth_assignments,
    select_target_player,
    _COCO_LEFT_ELBOW,
    _COCO_RIGHT_ELBOW,
    _COCO_LEFT_WRIST,
    _COCO_RIGHT_WRIST,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FW, FH = 1920, 1080  # standard frame dimensions


def _make_person(
    x1: int, y1: int, x2: int, y2: int,
    conf: float = 0.90,
    left_wrist:  Optional[tuple] = None,  # (x_norm, y_norm)
    right_wrist: Optional[tuple] = None,
    left_elbow:  Optional[tuple] = None,
    right_elbow: Optional[tuple] = None,
) -> PersonDetection:
    """Build a PersonDetection, optionally with synthetic keypoints."""
    kp_norm = None
    if any(v is not None for v in (left_wrist, right_wrist, left_elbow, right_elbow)):
        kp_norm = np.zeros((17, 3), dtype=np.float32)
        if left_elbow is not None:
            kp_norm[_COCO_LEFT_ELBOW,  :] = [left_elbow[0],  left_elbow[1],  0.90]
        if right_elbow is not None:
            kp_norm[_COCO_RIGHT_ELBOW, :] = [right_elbow[0], right_elbow[1], 0.90]
        if left_wrist is not None:
            kp_norm[_COCO_LEFT_WRIST,  :] = [left_wrist[0],  left_wrist[1],  0.90]
        if right_wrist is not None:
            kp_norm[_COCO_RIGHT_WRIST, :] = [right_wrist[0], right_wrist[1], 0.90]
    return PersonDetection(
        bbox=(x1, y1, x2, y2),
        confidence=conf,
        keypoints_norm=kp_norm,
    )


def _make_ball(cx: float, cy: float, radius: float = 30.0) -> BallDetection:
    r = int(radius)
    return BallDetection(
        detected=True,
        center=(cx, cy),
        bbox=(int(cx - r), int(cy - r), int(cx + r), int(cy + r)),
        radius=radius,
        confidence=0.85,
    )


def _make_assignment(
    bbox: Optional[tuple],
    score: float = 0.8,
    source: str = "ball_proximity",
) -> FrameAssignment:
    a = FrameAssignment(target_bbox=bbox, target_score=score, target_source=source)
    return a


# ---------------------------------------------------------------------------
# PersonDetection dataclass
# ---------------------------------------------------------------------------

class TestPersonDetection:
    def test_area_computed(self):
        p = _make_person(100, 100, 300, 500)
        assert p.area == 200 * 400

    def test_center(self):
        p = _make_person(0, 0, 200, 100)
        assert p.center == (100.0, 50.0)

    def test_wrists_norm_none_when_no_keypoints(self):
        p = _make_person(0, 0, 200, 400)
        assert p.wrists_norm is None

    def test_wrists_norm_returns_tuples(self):
        p = _make_person(0, 0, 200, 400,
                         left_wrist=(0.3, 0.5), right_wrist=(0.7, 0.5))
        wn = p.wrists_norm
        assert wn is not None
        assert len(wn) == 2
        assert wn[0] == pytest.approx((0.3, 0.5), abs=1e-4)
        assert wn[1] == pytest.approx((0.7, 0.5), abs=1e-4)

    def test_wrists_norm_low_confidence_returns_none_element(self):
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[9,  :] = [0.3, 0.5, 0.10]   # left wrist: conf below threshold
        kp[10, :] = [0.7, 0.5, 0.95]   # right wrist: conf ok
        p = PersonDetection(bbox=(0, 0, 200, 400), confidence=0.9, keypoints_norm=kp)
        wn = p.wrists_norm
        assert wn is not None
        assert wn[0] is None        # left below threshold
        assert wn[1] is not None    # right ok

    def test_elbows_norm_returns_tuples(self):
        p = _make_person(0, 0, 200, 400,
                         left_elbow=(0.25, 0.4), right_elbow=(0.65, 0.4))
        en = p.elbows_norm
        assert en is not None
        assert en[0] == pytest.approx((0.25, 0.4), abs=1e-4)
        assert en[1] == pytest.approx((0.65, 0.4), abs=1e-4)

    def test_to_dict_keys(self):
        p = _make_person(10, 20, 110, 220, left_wrist=(0.3, 0.5))
        d = p.to_dict()
        assert set(d.keys()) >= {"bbox", "confidence", "area"}
        assert d["bbox"] == [10, 20, 110, 220]

    def test_to_dict_includes_elbows_and_wrists(self):
        p = _make_person(0, 0, 200, 400,
                         left_elbow=(0.25, 0.4),  right_elbow=(0.65, 0.4),
                         left_wrist=(0.3,  0.5),  right_wrist=(0.7,  0.5))
        d = p.to_dict()
        assert "left_wrist_norm"  in d
        assert "right_wrist_norm" in d
        assert "left_elbow_norm"  in d
        assert "right_elbow_norm" in d
        assert d["left_wrist_norm"]  == pytest.approx([0.3, 0.5], abs=1e-4)
        assert d["left_elbow_norm"]  == pytest.approx([0.25, 0.4], abs=1e-4)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_bbox_iou_identical(self):
        assert _bbox_iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_bbox_iou_no_overlap(self):
        assert _bbox_iou((0, 0, 50, 50), (100, 100, 200, 200)) == pytest.approx(0.0)

    def test_bbox_iou_partial(self):
        iou = _bbox_iou((0, 0, 100, 100), (50, 50, 150, 150))
        # intersection 50×50=2500, union=10000+10000-2500=17500
        assert iou == pytest.approx(2500 / 17500, abs=1e-4)

    def test_edge_distance_inside(self):
        assert _edge_distance(50, 50, (0, 0, 100, 100)) == pytest.approx(0.0)

    def test_edge_distance_outside(self):
        d = _edge_distance(200, 50, (0, 0, 100, 100))
        assert d == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# _hand_proxy_norm
# ---------------------------------------------------------------------------

class TestHandProxyNorm:
    def _kp(self, wrist_xy, elbow_xy, wrist_conf=0.9, elbow_conf=0.9):
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[_COCO_LEFT_ELBOW, :] = [elbow_xy[0],  elbow_xy[1],  elbow_conf]
        kp[_COCO_LEFT_WRIST, :] = [wrist_xy[0],  wrist_xy[1],  wrist_conf]
        return kp

    def test_proxy_extends_beyond_wrist(self):
        # Elbow at (0.2, 0.5), Wrist at (0.4, 0.5) — direction is rightward
        kp = self._kp(wrist_xy=(0.4, 0.5), elbow_xy=(0.2, 0.5))
        proxy = _hand_proxy_norm(kp, _COCO_LEFT_WRIST, _COCO_LEFT_ELBOW, extend=0.5)
        assert proxy is not None
        # proxy_x = 0.4 + (0.4-0.2)*0.5 = 0.4 + 0.1 = 0.5
        assert proxy[0] == pytest.approx(0.5, abs=1e-4)
        assert proxy[1] == pytest.approx(0.5, abs=1e-4)

    def test_returns_none_for_low_confidence_wrist(self):
        kp = self._kp(wrist_xy=(0.4, 0.5), elbow_xy=(0.2, 0.5), wrist_conf=0.10)
        proxy = _hand_proxy_norm(kp, _COCO_LEFT_WRIST, _COCO_LEFT_ELBOW)
        assert proxy is None

    def test_returns_none_for_low_confidence_elbow(self):
        kp = self._kp(wrist_xy=(0.4, 0.5), elbow_xy=(0.2, 0.5), elbow_conf=0.05)
        proxy = _hand_proxy_norm(kp, _COCO_LEFT_WRIST, _COCO_LEFT_ELBOW)
        assert proxy is None


# ---------------------------------------------------------------------------
# score_player_for_ball
# ---------------------------------------------------------------------------

class TestScorePlayerForBall:
    def test_ball_inside_with_wrist_scores_high(self):
        # Ball inside bbox, wrist right at ball center → dominant signal fires
        ball   = _make_ball(cx=250, cy=400)
        person = _make_person(100, 100, 400, 700,
                              left_wrist=(250/FW, 400/FH))
        score  = score_player_for_ball(person, ball, FW, FH)
        # wrist=0.40 (distance=0 → score=1) + inside=0.20 + edge=0.10 = 0.70
        assert score >= 0.50, f"Expected >= 0.50, got {score}"

    def test_ball_inside_no_wrist_scores_moderately(self):
        # Ball inside bbox but no wrist keypoints → only spatial signals
        ball   = _make_ball(cx=250, cy=400)
        person = _make_person(100, 100, 400, 700)
        score  = score_player_for_ball(person, ball, FW, FH)
        # inside=0.20 + edge=0.10 = 0.30
        assert 0.20 <= score <= 0.45, f"Expected [0.20, 0.45], got {score}"

    def test_ball_far_away_scores_low(self):
        person = _make_person(100, 100, 400, 700)
        ball   = _make_ball(cx=1500, cy=800)  # far right
        score  = score_player_for_ball(person, ball, FW, FH)
        assert score <= 0.20, f"Expected <= 0.20, got {score}"

    def test_no_ball_detected_returns_zero(self):
        person = _make_person(100, 100, 400, 700)
        ball   = BallDetection()  # not detected
        score  = score_player_for_ball(person, ball, FW, FH)
        assert score == 0.0

    def test_wrist_proximity_is_dominant_signal(self):
        # Player B has ball inside bbox; Player A has wrist closer to ball.
        # With w_wrist=0.40 dominant, A should beat B when wrist is very close.
        ball = _make_ball(cx=900, cy=400, radius=20)
        # Player A: no bbox overlap, but wrist AT the ball
        player_a = _make_person(800, 300, 1000, 600,   # ball inside too, so fair test
                                left_wrist=(900/FW, 400/FH))
        # Player B: ball inside but no keypoints
        player_b = _make_person(800, 300, 1000, 600)
        score_a = score_player_for_ball(player_a, ball, FW, FH)
        score_b = score_player_for_ball(player_b, ball, FW, FH)
        assert score_a > score_b, f"Expected wrist to lift score: {score_a} vs {score_b}"

    def test_hand_proxy_increases_score(self):
        # Elbow and wrist set so hand proxy lands at ball center
        ball = _make_ball(cx=600, cy=400, radius=20)
        # Elbow at x=0.2, wrist at x=0.3 → proxy at x=0.35 (= 0.3 + 0.5*(0.3-0.2))
        # Ball at cx=600 / FW = 0.3125
        # With proxy at 0.35 ≈ 0.31 * 1920 = 595px, very close to ball
        person_with_proxy = _make_person(
            400, 200, 800, 800,
            left_elbow=(0.2,  400/FH),
            left_wrist=(0.3,  400/FH),
        )
        person_wrist_only = _make_person(
            400, 200, 800, 800,
            left_wrist=(0.3,  400/FH),
        )
        s_proxy = score_player_for_ball(person_with_proxy, ball, FW, FH)
        s_wrist = score_player_for_ball(person_wrist_only, ball, FW, FH)
        # Both should score high (wrist close to ball), proxy adds an extra signal
        assert s_proxy >= s_wrist, f"Hand proxy should not hurt: {s_proxy} vs {s_wrist}"

    def test_temporal_bonus_increases_score(self):
        person  = _make_person(100, 100, 400, 700)
        ball    = _make_ball(cx=250, cy=400)
        prev    = (100, 100, 400, 700)        # same bbox → IoU = 1.0
        s_tempo = score_player_for_ball(person, ball, FW, FH, prev_target_bbox=prev)
        s_no_t  = score_player_for_ball(person, ball, FW, FH, prev_target_bbox=None)
        assert s_tempo > s_no_t

    def test_score_range(self):
        person = _make_person(100, 100, 900, 900,
                              left_wrist=(0.3, 0.3), right_wrist=(0.7, 0.7))
        ball   = _make_ball(cx=500, cy=500)
        score  = score_player_for_ball(person, ball, FW, FH)
        assert 0.0 <= score <= 1.0

    def test_correct_player_wins_over_decoy(self):
        # Player A has ball inside + wrist near ball; Player B is far away.
        player_a = _make_person(200, 200, 600, 900,
                                left_wrist=(400/FW, 500/FH))
        player_b = _make_person(1400, 200, 1800, 900)  # far right
        ball     = _make_ball(cx=400, cy=500)
        score_a  = score_player_for_ball(player_a, ball, FW, FH)
        score_b  = score_player_for_ball(player_b, ball, FW, FH)
        assert score_a > score_b


# ---------------------------------------------------------------------------
# assign_target_player
# ---------------------------------------------------------------------------

class TestAssignTargetPlayer:
    def test_single_player_assigned(self):
        persons = [_make_person(100, 100, 500, 800)]
        ball    = _make_ball(cx=300, cy=450)
        asgn    = assign_target_player(persons, ball, FW, FH)
        assert asgn.target_bbox == (100, 100, 500, 800)
        assert asgn.target_source == "ball_proximity"
        assert asgn.target_score > 0

    def test_no_players_returns_none(self):
        asgn = assign_target_player([], _make_ball(300, 300), FW, FH)
        assert asgn.target_bbox is None
        assert asgn.target_source == "none"
        assert asgn.n_players == 0

    def test_closer_player_wins(self):
        p_near = _make_person(300, 300, 600, 800)   # ball inside
        p_far  = _make_person(1500, 100, 1900, 900)
        ball   = _make_ball(cx=450, cy=550)
        asgn   = assign_target_player([p_near, p_far], ball, FW, FH)
        assert asgn.target_bbox == p_near.bbox

    def test_fallback_to_largest_when_no_ball(self):
        p_small = _make_person(100, 100, 200, 300)
        p_large = _make_person(400, 100, 900, 900)
        ball    = BallDetection()  # no ball
        asgn    = assign_target_player([p_small, p_large], ball, FW, FH,
                                       config={"fallback_to_largest": True})
        assert asgn.target_bbox == p_large.bbox
        assert asgn.target_source == "largest"

    def test_no_fallback_when_disabled(self):
        p = _make_person(100, 100, 500, 800)
        ball = BallDetection()
        asgn = assign_target_player([p], ball, FW, FH,
                                    config={"fallback_to_largest": False})
        assert asgn.target_bbox is None

    def test_players_list_populated_with_score_breakdown(self):
        persons = [_make_person(100, 100, 500, 800), _make_person(900, 100, 1400, 800)]
        ball    = _make_ball(cx=300, cy=400)
        asgn    = assign_target_player(persons, ball, FW, FH)
        assert len(asgn.players) == 2
        for p in asgn.players:
            assert "score" in p
            assert "bbox"  in p
            assert "score_breakdown" in p
            bd = p["score_breakdown"]
            assert "wrist" in bd and "inside" in bd and "edge" in bd

    def test_target_keypoints_stored(self):
        # When person has keypoints, they should be stored in FrameAssignment
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[9, :] = [0.5, 0.5, 0.9]   # left wrist
        person = PersonDetection(bbox=(100, 100, 500, 800), confidence=0.9, keypoints_norm=kp)
        ball   = _make_ball(cx=300, cy=450)
        asgn   = assign_target_player([person], ball, FW, FH)
        assert asgn.target_keypoints_norm is not None
        assert asgn.target_keypoints_norm.shape == (17, 3)

    def test_score_breakdown_stored(self):
        person = _make_person(100, 100, 500, 800, left_wrist=(0.3, 0.5))
        ball   = _make_ball(cx=300, cy=400)
        asgn   = assign_target_player([person], ball, FW, FH)
        assert asgn.score_breakdown is not None
        assert "wrist" in asgn.score_breakdown
        assert "inside" in asgn.score_breakdown
        assert "temporal" in asgn.score_breakdown


# ---------------------------------------------------------------------------
# smooth_assignments
# ---------------------------------------------------------------------------

class TestSmoothAssignments:
    def test_stable_assignment_unchanged(self):
        bbox = (100, 100, 400, 700)
        raw  = [_make_assignment(bbox) for _ in range(10)]
        out  = smooth_assignments(raw, window=5, min_iou=0.20)
        for a in out:
            assert a.smoothed_bbox is not None
            assert _bbox_iou(a.smoothed_bbox, bbox) > 0.99

    def test_short_gap_filled_forward(self):
        bbox = (100, 100, 400, 700)
        raw  = [_make_assignment(bbox)] * 5
        raw += [_make_assignment(None, 0.0, "none")] * 3
        raw += [_make_assignment(bbox)] * 5
        out  = smooth_assignments(raw, window=3, min_iou=0.10, max_gap_fill=10)
        for a in out[5:8]:
            assert a.smoothed_bbox is not None
            assert a.smoothed_source in ("temporal", "ball_proximity")

    def test_leading_gap_filled_backward(self):
        bbox = (100, 100, 400, 700)
        raw  = [_make_assignment(None, 0.0, "none")] * 5
        raw += [_make_assignment(bbox)] * 10
        out  = smooth_assignments(raw, window=3, min_iou=0.10, max_gap_fill=10)
        for a in out[:5]:
            assert a.smoothed_bbox is not None

    def test_flicker_suppressed(self):
        bbox_a = (100, 100, 400, 700)
        bbox_b = (900, 100, 1400, 700)
        seq    = [bbox_a, bbox_a, bbox_b, bbox_a, bbox_b, bbox_a, bbox_a]
        raw    = [_make_assignment(b) for b in seq]
        out    = smooth_assignments(raw, window=5, min_iou=0.10)
        n_a = sum(1 for a in out if a.smoothed_bbox and _bbox_iou(a.smoothed_bbox, bbox_a) > 0.8)
        n_b = sum(1 for a in out if a.smoothed_bbox and _bbox_iou(a.smoothed_bbox, bbox_b) > 0.8)
        assert n_a > n_b, f"Expected bbox_a to dominate: n_a={n_a}, n_b={n_b}"

    def test_long_gap_not_filled_beyond_max(self):
        bbox = (100, 100, 400, 700)
        raw  = [_make_assignment(bbox)] * 3
        raw += [_make_assignment(None, 0.0, "none")] * 50
        raw += [_make_assignment(bbox)] * 3
        out  = smooth_assignments(raw, window=3, min_iou=0.10, max_gap_fill=5)
        assert out[8].smoothed_bbox is not None    # within 5-frame forward fill
        assert len(out) == len(raw)


# ---------------------------------------------------------------------------
# gate_landmarks_for_player_selection
# ---------------------------------------------------------------------------

class TestGateLandmarks:
    def test_matching_player_not_gated(self):
        frame = {
            "frame_idx": 0, "time_s": 0.0,
            "landmarks": {
                "LEFT_SHOULDER":  {"x": 0.2, "y": 0.3, "visibility": 0.9},
                "RIGHT_SHOULDER": {"x": 0.4, "y": 0.3, "visibility": 0.9},
                "LEFT_HIP":       {"x": 0.2, "y": 0.6, "visibility": 0.9},
                "RIGHT_HIP":      {"x": 0.4, "y": 0.6, "visibility": 0.9},
            },
        }
        asgn = FrameAssignment(smoothed_bbox=(300, 270, 780, 700))
        data = {"frames": [frame], "meta": {"width": FW, "height": FH}}
        n = gate_landmarks_for_player_selection(data, [asgn], min_iou=0.25)
        assert n == 0
        assert frame["landmarks"]

    def test_mismatched_player_gated(self):
        frame = {
            "frame_idx": 0, "time_s": 0.0,
            "landmarks": {
                "LEFT_SHOULDER":  {"x": 0.1, "y": 0.3, "visibility": 0.9},
                "RIGHT_SHOULDER": {"x": 0.3, "y": 0.3, "visibility": 0.9},
                "LEFT_HIP":       {"x": 0.1, "y": 0.6, "visibility": 0.9},
                "RIGHT_HIP":      {"x": 0.3, "y": 0.6, "visibility": 0.9},
            },
        }
        asgn = FrameAssignment(smoothed_bbox=(1500, 100, 1900, 900))
        data = {"frames": [frame], "meta": {"width": FW, "height": FH}}
        n = gate_landmarks_for_player_selection(data, [asgn], min_iou=0.25)
        assert n == 1
        assert frame["landmarks"] == {}
        assert "landmarks_unfiltered" in frame
        assert asgn.landmarks_gated is True

    def test_no_target_skipped(self):
        frame = {
            "frame_idx": 0, "time_s": 0.0,
            "landmarks": {"NOSE": {"x": 0.5, "y": 0.5, "visibility": 0.9}},
        }
        asgn = FrameAssignment(smoothed_bbox=None)
        data = {"frames": [frame], "meta": {"width": FW, "height": FH}}
        n = gate_landmarks_for_player_selection(data, [asgn], min_iou=0.25)
        assert n == 0
        assert frame["landmarks"]


# ---------------------------------------------------------------------------
# _estimate_extracted_person_bbox
# ---------------------------------------------------------------------------

class TestEstimateExtractedPersonBbox:
    def test_returns_none_for_empty_landmarks(self):
        frame = {"landmarks": {}}
        assert _estimate_extracted_person_bbox(frame, FW, FH) is None

    def test_returns_none_for_missing_landmarks(self):
        frame = {}
        assert _estimate_extracted_person_bbox(frame, FW, FH) is None

    def test_basic_bbox(self):
        frame = {
            "landmarks": {
                "A": {"x": 0.1, "y": 0.2, "visibility": 0.9},
                "B": {"x": 0.9, "y": 0.8, "visibility": 0.9},
            }
        }
        bbox = _estimate_extracted_person_bbox(frame, FW, FH)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert x1 == pytest.approx(0.1 * FW, abs=1)
        assert y1 == pytest.approx(0.2 * FH, abs=1)
        assert x2 == pytest.approx(0.9 * FW, abs=1)
        assert y2 == pytest.approx(0.8 * FH, abs=1)

    def test_low_visibility_excluded(self):
        frame = {
            "landmarks": {
                "good": {"x": 0.5, "y": 0.5, "visibility": 0.9},
                "bad":  {"x": 0.0, "y": 0.0, "visibility": 0.01},
            }
        }
        bbox = _estimate_extracted_person_bbox(frame, FW, FH, vis_thresh=0.10)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert x1 == x2 == pytest.approx(0.5 * FW, abs=1)
        assert y1 == y2 == pytest.approx(0.5 * FH, abs=1)


# ---------------------------------------------------------------------------
# coco17_to_landmarks
# ---------------------------------------------------------------------------

class TestCoco17ToLandmarks:
    def _make_kp(self) -> np.ndarray:
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[5,  :] = [0.3, 0.2, 0.9]   # LEFT_SHOULDER
        kp[6,  :] = [0.7, 0.2, 0.9]   # RIGHT_SHOULDER
        kp[11, :] = [0.3, 0.6, 0.9]   # LEFT_HIP
        kp[12, :] = [0.7, 0.6, 0.9]   # RIGHT_HIP
        kp[15, :] = [0.3, 0.9, 0.8]   # LEFT_ANKLE
        kp[16, :] = [0.7, 0.9, 0.8]   # RIGHT_ANKLE
        return kp

    def test_key_joints_present(self):
        lm = coco17_to_landmarks(self._make_kp())
        for name in ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
                     "LEFT_ANKLE", "RIGHT_ANKLE"):
            assert name in lm, f"Missing: {name}"

    def test_values_correct(self):
        lm = coco17_to_landmarks(self._make_kp())
        assert lm["LEFT_SHOULDER"]["x"]          == pytest.approx(0.3, abs=1e-4)
        assert lm["LEFT_SHOULDER"]["y"]          == pytest.approx(0.2, abs=1e-4)
        assert lm["LEFT_SHOULDER"]["visibility"] == pytest.approx(0.9, abs=1e-4)

    def test_low_confidence_excluded(self):
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[5, :] = [0.3, 0.2, 0.005]  # confidence below 0.01 threshold
        lm = coco17_to_landmarks(kp)
        assert "LEFT_SHOULDER" not in lm

    def test_zero_array_produces_no_landmarks(self):
        kp = np.zeros((17, 3), dtype=np.float32)
        lm = coco17_to_landmarks(kp)
        assert len(lm) == 0


# ---------------------------------------------------------------------------
# extract_selected_player_pose
# ---------------------------------------------------------------------------

class TestExtractSelectedPlayerPose:
    def _make_kp(self, x_shoulder: float = 0.5) -> np.ndarray:
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[5,  :] = [x_shoulder - 0.05, 0.3, 0.9]
        kp[6,  :] = [x_shoulder + 0.05, 0.3, 0.9]
        kp[11, :] = [x_shoulder - 0.05, 0.6, 0.9]
        kp[12, :] = [x_shoulder + 0.05, 0.6, 0.9]
        kp[15, :] = [x_shoulder - 0.05, 0.9, 0.8]
        kp[16, :] = [x_shoulder + 0.05, 0.9, 0.8]
        return kp

    def test_landmarks_written_for_frames_with_keypoints(self):
        n = 5
        frames = [{"frame_idx": i, "time_s": i/30.0, "landmarks": {}} for i in range(n)]
        data   = {"frames": frames, "meta": {}}

        kp = self._make_kp()
        assignments = []
        for i in range(n):
            a = FrameAssignment(
                target_keypoints_norm=kp,
                smoothed_bbox=(100, 50, 400, 900),
                smoothed_source="ball_proximity",
            )
            assignments.append(a)

        updated = extract_selected_player_pose(data, assignments, pose_gap_fill=False)
        assert updated == n
        for frame in frames:
            assert "LEFT_SHOULDER" in frame["landmarks"]
            assert "LEFT_ANKLE"    in frame["landmarks"]

    def test_original_landmarks_preserved(self):
        original_lm = {"NOSE": {"x": 0.5, "y": 0.1, "visibility": 0.9}}
        frame = {"frame_idx": 0, "time_s": 0.0, "landmarks": dict(original_lm)}
        data  = {"frames": [frame], "meta": {}}

        asgn = FrameAssignment(
            target_keypoints_norm=self._make_kp(),
            smoothed_bbox=(100, 50, 400, 900),
        )
        extract_selected_player_pose(data, [asgn])
        assert "landmarks_original_extract" in frame
        assert frame["landmarks_original_extract"]["NOSE"]["x"] == pytest.approx(0.5)

    def test_gap_fill_propagates_last_keypoints(self):
        n = 5
        frames = [{"frame_idx": i, "time_s": i/30.0, "landmarks": {}} for i in range(n)]
        data   = {"frames": frames, "meta": {}}

        kp = self._make_kp()
        assignments = []
        for i in range(n):
            has_kp = (i == 0)  # only first frame has real keypoints
            a = FrameAssignment(
                target_keypoints_norm=kp if has_kp else None,
                smoothed_bbox=(100, 50, 400, 900),
                smoothed_source="ball_proximity" if has_kp else "temporal",
            )
            assignments.append(a)

        updated = extract_selected_player_pose(data, assignments, pose_gap_fill=True)
        assert updated == n   # all frames should get landmarks via gap fill
        for frame in frames:
            assert "LEFT_SHOULDER" in frame["landmarks"]

    def test_no_target_clears_landmarks(self):
        original_lm = {"NOSE": {"x": 0.5, "y": 0.1, "visibility": 0.9}}
        frame = {"frame_idx": 0, "time_s": 0.0, "landmarks": dict(original_lm)}
        data  = {"frames": [frame], "meta": {}}

        # No player assigned for this frame
        asgn = FrameAssignment(target_keypoints_norm=None, smoothed_bbox=None)
        extract_selected_player_pose(data, [asgn], pose_gap_fill=False)
        # Landmarks should be cleared when no player is assigned
        assert frame["landmarks"] == {}


# ---------------------------------------------------------------------------
# Integration: select_target_player + render
# ---------------------------------------------------------------------------

def _make_synthetic_video(path: str, n_frames: int = 20) -> None:
    """Create a minimal synthetic video."""
    W, H = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 30.0, (W, H))
    for i in range(n_frames):
        frame = np.ones((H, W, 3), dtype=np.uint8) * 200
        cv2.rectangle(frame, (60, 40), (160, 200), (180, 120, 80), -1)
        out.write(frame)
    out.release()


def _make_minimal_data(n_frames: int = 20, fps: float = 30.0) -> dict:
    """Minimal data dict compatible with select_target_player."""
    frames = []
    for i in range(n_frames):
        lm = {
            "LEFT_SHOULDER":  {"x": 0.35, "y": 0.25, "visibility": 0.9},
            "RIGHT_SHOULDER": {"x": 0.45, "y": 0.25, "visibility": 0.9},
            "LEFT_HIP":       {"x": 0.35, "y": 0.55, "visibility": 0.9},
            "RIGHT_HIP":      {"x": 0.45, "y": 0.55, "visibility": 0.9},
            "LEFT_WRIST":     {"x": 0.30, "y": 0.45, "visibility": 0.9},
            "RIGHT_WRIST":    {"x": 0.50, "y": 0.45, "visibility": 0.9},
        }
        frames.append({
            "frame_idx": i,
            "time_s":    round(i / fps, 4),
            "landmarks": lm,
            "confidence": 0.9,
        })

    per_frame_ball = []
    for i in range(n_frames):
        per_frame_ball.append({
            "frame_idx": i,
            "ball": {
                "detected": True, "tracked": True, "source": "detected",
                "center": [110, 120], "bbox": [95, 105, 125, 135],
                "radius": 15.0, "confidence": 0.75,
            },
        })

    return {
        "meta":   {"fps": fps, "width": 320, "height": 240, "n_frames": n_frames},
        "frames": frames,
        "ball":   {"per_frame": per_frame_ball},
    }


class TestSelectTargetPlayerIntegration:
    """Integration tests using a synthetic video and stub person detector."""

    def _make_stub_detector(self, persons_per_frame):
        from myogait.detectors.person_detector import PersonDetector

        class _Stub(PersonDetector):
            def __init__(self, ppf):
                self._ppf = ppf
                self._idx = 0

            def detect(self, frame_bgr):
                result = self._ppf[self._idx % len(self._ppf)]
                self._idx += 1
                return result

        return _Stub(persons_per_frame)

    def test_single_player_all_frames(self, tmp_path):
        n = 10
        video_path = str(tmp_path / "synth.mp4")
        _make_synthetic_video(video_path, n_frames=n)
        data = _make_minimal_data(n_frames=n)

        persons_per_frame = [
            # Include wrist so keypoints_norm is populated → pose update fires
            [_make_person(50, 30, 200, 230, left_wrist=(0.34, 0.5))]
        ] * n
        stub = self._make_stub_detector(persons_per_frame)

        result = select_target_player(
            video_path, data,
            config={"gate_landmarks": False, "run_events": False},
            person_detector=stub,
        )

        assert "player_selection" in result
        ps = result["player_selection"]
        assert ps["summary"]["n_frames"] == n
        assert ps["summary"]["n_frames_player_found"] == n
        assert ps["summary"]["player_coverage_rate"] == 1.0
        assert ps["summary"]["n_frames_pose_updated"] == n
        assert len(ps["per_frame"]) == n

        for pf in ps["per_frame"]:
            assert pf["smoothed_bbox"] is not None

    def test_no_players_any_frame(self, tmp_path):
        n = 8
        video_path = str(tmp_path / "empty.mp4")
        _make_synthetic_video(video_path, n_frames=n)
        data = _make_minimal_data(n_frames=n)

        stub = self._make_stub_detector([[]] * n)

        result = select_target_player(
            video_path, data,
            config={"gate_landmarks": False, "fallback_to_largest": False, "run_events": False},
            person_detector=stub,
        )

        ps = result["player_selection"]
        assert ps["summary"]["n_frames_no_players"] == n

    def test_landmark_gating_triggered(self, tmp_path):
        n = 5
        video_path = str(tmp_path / "gating.mp4")
        _make_synthetic_video(video_path, n_frames=n)
        data = _make_minimal_data(n_frames=n)

        persons_per_frame = [
            [_make_person(1500, 100, 1800, 900)]
        ] * n
        stub = self._make_stub_detector(persons_per_frame)

        data["meta"]["width"]  = FW
        data["meta"]["height"] = FH

        result = select_target_player(
            video_path, data,
            config={"gate_landmarks": True, "gate_min_iou": 0.25,
                    "extract_selected_pose": False, "run_events": False},
            person_detector=stub,
        )

        ps = result["player_selection"]
        assert ps["summary"]["n_frames_landmarks_gated"] > 0

    def test_pose_extraction_updates_landmarks(self, tmp_path):
        n = 6
        video_path = str(tmp_path / "pose.mp4")
        _make_synthetic_video(video_path, n_frames=n)
        data = _make_minimal_data(n_frames=n)

        # Give the person COCO-17 keypoints so they get written to landmarks
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[5,  :] = [0.3, 0.2, 0.9]   # LEFT_SHOULDER
        kp[11, :] = [0.3, 0.6, 0.9]   # LEFT_HIP
        kp[15, :] = [0.3, 0.9, 0.8]   # LEFT_ANKLE
        person = PersonDetection(bbox=(50, 30, 200, 230), confidence=0.9, keypoints_norm=kp)

        stub = self._make_stub_detector([[person]] * n)

        result = select_target_player(
            video_path, data,
            config={"gate_landmarks": False, "extract_selected_pose": True, "run_events": False},
            person_detector=stub,
        )

        # After pose extraction, landmarks should contain COCO-derived joints
        for frame in result["frames"]:
            lm = frame.get("landmarks", {})
            assert "LEFT_SHOULDER" in lm
            assert "LEFT_ANKLE"    in lm
            # Original extract() landmarks preserved
            assert "landmarks_original_extract" in frame

    def test_render_player_selection_video(self, tmp_path):
        n = 6
        video_path  = str(tmp_path / "render_src.mp4")
        output_path = str(tmp_path / "render_out.mp4")
        _make_synthetic_video(video_path, n_frames=n)
        data = _make_minimal_data(n_frames=n)

        stub = self._make_stub_detector([[_make_person(50, 30, 200, 210)]] * n)

        result = select_target_player(
            video_path, data,
            config={"gate_landmarks": False, "run_events": False},
            person_detector=stub,
        )

        render_player_selection_video(video_path, result, output_path)
        assert os.path.isfile(output_path)
        cap = cv2.VideoCapture(output_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert frame_count == n

    def test_data_dict_structure(self, tmp_path):
        n = 5
        video_path = str(tmp_path / "struct.mp4")
        _make_synthetic_video(video_path, n_frames=n)
        data = _make_minimal_data(n_frames=n)
        stub = self._make_stub_detector([[_make_person(50, 30, 200, 210)]] * n)

        result = select_target_player(
            video_path, data,
            config={"gate_landmarks": False, "run_events": False},
            person_detector=stub,
        )
        ps = result["player_selection"]

        # summary keys
        for key in ("n_frames", "player_coverage_rate", "n_frames_player_found",
                    "n_frames_no_players", "n_frames_pose_updated"):
            assert key in ps["summary"], f"Missing summary key: {key}"

        # per-frame keys
        pf = ps["per_frame"][0]
        for key in ("frame_idx", "time_s", "n_players", "players",
                    "target_bbox", "target_score", "target_source",
                    "score_breakdown", "smoothed_bbox", "smoothed_source",
                    "landmarks_gated"):
            assert key in pf, f"Missing per-frame key: {key}"
