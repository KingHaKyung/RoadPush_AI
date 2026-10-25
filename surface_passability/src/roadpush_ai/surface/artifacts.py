from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from roadpush_ai.validation.surface_evaluator import contour, paint
from roadpush_ai.validation.frozen_suite import (
    CYAN,
    DARK_BLUE,
    GRAY,
    GREEN,
    MAGENTA,
    ORANGE,
    YELLOW,
    _applied_blocker_panel,
    _draw_path,
    _obstacle_panel,
    _record_source_mask,
    _row_panel,
    review_relevant_blockers,
)


def mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
    ).hexdigest()


def obstacle_sources(pipeline, blockers: dict) -> list[dict]:
    sources = []
    for index, record in enumerate(blockers["objects"]):
        usable = bool(record.get("accepted")) or record.get("kind") == "transient_occluder"
        if not usable:
            continue
        mask = _record_source_mask(record, pipeline.labels)
        if not mask.any():
            continue
        sources.append(
            {
                "index": index,
                "mask": mask,
                "area": int(mask.sum()),
                "bbox": record.get("bbox"),
                "class_name": record.get("class_name"),
                "kind": record.get("kind"),
                "strategy": record.get("strategy"),
            }
        )
    return sources


def _component_panel(image: np.ndarray, selection: dict) -> np.ndarray:
    panel = paint(image, selection["rejected_sources"], GRAY, 0.62)
    panel = paint(panel, selection["main"], GREEN, 0.42)
    panel = paint(panel, selection["accepted_satellites"], CYAN, 0.70)
    panel = contour(panel, selection["main"], MAGENTA, 2)
    if selection["accepted_satellites"].any():
        panel = contour(panel, selection["accepted_satellites"], CYAN, 2)
    return panel


def render_sidewalk_review(
    artifact: Path,
    sample_id: str,
    pipeline,
    current_stage,
    proposed_stage,
    selection: dict,
    blockers: dict,
    evaluated: dict,
) -> None:
    p1 = _component_panel(pipeline.image, selection)
    p2 = contour(
        paint(pipeline.image, current_stage.raw_boundary, GREEN, 0.38),
        current_stage.raw_boundary,
        MAGENTA,
        2,
    )
    p3 = contour(
        paint(pipeline.image, proposed_stage.raw_boundary, GREEN, 0.38),
        proposed_stage.raw_boundary,
        MAGENTA,
        2,
    )
    _draw_path(p3, pipeline.route_path, DARK_BLUE, 5)
    _draw_path(p3, proposed_stage.route_path, YELLOW, 4)
    p4 = pipeline.image.copy()
    p4 = paint(p4, proposed_stage.vetoed_candidate, GRAY, 0.78)
    p4 = paint(
        p4,
        proposed_stage.raw_candidate & ~proposed_stage.vetoed_candidate,
        ORANGE,
        0.78,
    )
    p5 = contour(
        paint(pipeline.image, proposed_stage.pruned_boundary, GREEN, 0.38),
        proposed_stage.pruned_boundary,
        MAGENTA,
        2,
    )
    p5 = paint(p5, proposed_stage.approved_removal, ORANGE, 0.82)
    review_blockers, _ = review_relevant_blockers(
        pipeline.labels,
        proposed_stage.pruned_boundary,
        blockers,
        evaluated["rows"],
    )
    p6 = _obstacle_panel(pipeline.image, review_blockers)
    p7 = _applied_blocker_panel(
        pipeline.image, proposed_stage.pruned_boundary, blockers
    )
    p8 = contour(
        paint(pipeline.image, evaluated["free"], GREEN, 0.44),
        evaluated["free"],
        MAGENTA,
        3,
    )
    p9 = _row_panel(pipeline.image, evaluated)
    panels = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
    titles = [
        "1 Component selection: green=main, cyan=obstacle-linked, gray=rejected",
        "2 CURRENT Test3 all-candidate outer row span",
        "3 PROPOSED selected-component span + routes",
        "4 Proposed side candidates: orange=accepted, gray=veto",
        "5 Proposed boundary after side pruning",
        "6 Relevant segmentation + grounded-all banner grouping",
        "7 Applied blocker inside proposed boundary",
        "8 Proposed final free sidewalk + true final boundary",
        "9 Proposed passable width: every post-blocker free segment",
    ]
    fig, axes = plt.subplots(
        3, 3, figsize=(17, 15), constrained_layout=True, facecolor="white"
    )
    for axis, title, panel in zip(axes.flat, titles, panels):
        axis.imshow(panel)
        axis.set_title(title, fontsize=8.7, color="black")
        axis.axis("off")
    axes.flat[0].legend(
        handles=[
            Patch(color="#2dd25f", label="main observed component"),
            Patch(color="#28c8ff", label="accepted obstacle-linked satellite"),
            Patch(color="#919191", label="rejected candidate component"),
        ],
        loc="lower left",
        fontsize=6.8,
        framealpha=0.82,
    )
    fig.suptitle(
        f"{sample_id} | obstacle-connected component probe | "
        f"accepted satellites {len(selection['evidence'])} | "
        f"current/proposed boundary {int(current_stage.raw_boundary.sum()):,}/"
        f"{int(proposed_stage.raw_boundary.sum()):,}px",
        fontsize=12.5,
        color="black",
    )
    fig.savefig(artifact / "summary.png", dpi=112, facecolor="white")
    plt.close(fig)
