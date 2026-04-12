import inspect
from pathlib import Path

import cv2
import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment
from restoration.candidates import ConnectionCandidate
from restoration.extraction import EndpointInfo, ExtractionResult
from restoration.pipeline import _build_report, _save_labeled_restoration


def _line_segment(p0, p3, source_type="contour"):
    p0 = np.asarray(p0, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    d = p3 - p0
    cp = np.vstack([p0, p0 + d / 3.0, p0 + 2.0 * d / 3.0, p3])
    return BezierSegment(control_points=cp, source_type=source_type)


def _endpoint(endpoint_id, path_index, end, position, tangent):
    return EndpointInfo(
        endpoint_id=endpoint_id,
        path_index=path_index,
        end=end,
        position=np.asarray(position, dtype=np.float64),
        tangent=np.asarray(tangent, dtype=np.float64),
        curvature=0.0,
    )


def test_overlay_event_includes_detailed_bridge_metadata(tmp_path: Path):
    img = np.full((240, 320, 3), 220, dtype=np.uint8)
    image_path = tmp_path / "mini.png"
    assert cv2.imwrite(str(image_path), img)

    bridge_seg = _line_segment((80.0, 120.0), (180.0, 120.0), source_type="bridge")
    final_path = BezierPath(segments=[bridge_seg], is_closed=False, source_type="restored")

    ep_a = _endpoint(0, 0, "end", (80.0, 120.0), (1.0, 0.0))
    ep_b = _endpoint(1, 1, "start", (180.0, 120.0), (-1.0, 0.0))
    candidate = ConnectionCandidate(
        id=7,
        ep_a=ep_a,
        ep_b=ep_b,
        scenario="continuation",
        bridge_points=bridge_seg.sample(16),
        bridge_bezier=[bridge_seg],
        distance=100.0,
        tier=1,
        bilateral_alignment=0.92,
        misalignment_deg=8.0,
        spur_involved=False,
        same_path_closure=False,
        score=0.81,
    )

    out_path, events = _save_labeled_restoration(
        str(image_path),
        [final_path],
        [candidate],
        str(tmp_path),
    )

    assert out_path
    assert Path(out_path).exists()
    assert len(events) == 1
    event = events[0]
    assert event["source"] == "asp_bridge"
    assert event["candidate"]["id"] == 7
    assert "candidate C7" in event["explanation"]
    assert "segment_count" in event
    assert "geometry" in event


def test_report_schema_is_detailed_and_restructured():
    seg = _line_segment((10.0, 20.0), (110.0, 20.0), source_type="bridge")
    path = BezierPath(segments=[seg], is_closed=False, source_type="restored")

    ep_a = _endpoint(2, 0, "end", (10.0, 20.0), (1.0, 0.0))
    ep_b = _endpoint(3, 1, "start", (110.0, 20.0), (-1.0, 0.0))

    candidate = ConnectionCandidate(
        id=9,
        ep_a=ep_a,
        ep_b=ep_b,
        scenario="extension_intersection",
        bridge_points=seg.sample(12),
        bridge_bezier=[seg],
        distance=100.0,
        tier=2,
        bilateral_alignment=0.66,
        misalignment_deg=21.0,
        spur_involved=True,
        same_path_closure=False,
        score=0.52,
    )

    extraction = ExtractionResult(
        paths=[path],
        endpoints=[ep_a, ep_b],
        efd_contours=[],
        image_shape=(240, 320),
        diagonal=float(np.hypot(240.0, 320.0)),
    )

    event = {
        "id": "R1",
        "source": "asp_bridge",
        "type": "ASP Junction Bridge",
        "coordinates": [60.0, 20.0],
        "segment_count": 1,
        "geometry": {"approx_length_px": 100.0, "sample_points": 12},
        "candidate": {"id": 9},
        "explanation": "Detailed explanation text",
    }

    report = _build_report(
        "synthetic.png",
        extraction,
        [candidate],
        [candidate],
        [path],
        0.321,
        restoration_history=[event],
        dropped_after_sanitize=0,
    )

    assert report["schema_version"] == "2.0"
    assert report["image"]["name"] == "synthetic.png"
    assert "summary" in report
    assert "analysis" in report
    assert "detailed_events" in report
    assert report["analysis"]["selection"]["accepted_candidate_ids"] == [9]
    assert report["analysis"]["candidate_generation"]["scenario_distribution"]["extension_intersection"] == 1


def test_overlay_source_no_longer_uses_glow_style_literals():
    source = inspect.getsource(_save_labeled_restoration)
    assert "#00FF00" not in source
    assert "#39FF14" not in source
    assert "fontsize=10" not in source
