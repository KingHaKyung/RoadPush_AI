"""Conditional vehicle-instance inference primitives."""

from __future__ import annotations

import cv2
import numpy as np
import os
from pathlib import Path
from threading import Lock
from PIL import Image, ImageDraw
from typing import Any, Mapping, Protocol, Sequence

from surface_passability.contracts.obstacles import (
    VehicleInstance,
    VehicleInstanceMatch,
    VehicleInstanceMatchEvidence,
    VehicleInstancePartition,
    VehicleInferenceDiagnostics,
    VehicleInferenceResult,
)
from surface_passability.obstacles.taxonomy import VEHICLE_INSTANCE_CLASS_NAMES
from surface_passability.policies.obstacles import (
    VEHICLE_INFERENCE_CONFIDENCE_THRESHOLD,
    VEHICLE_INFERENCE_DEFAULT_DEVICE,
    VEHICLE_INFERENCE_DEVICE_ENVIRONMENT_VARIABLE,
    VEHICLE_INFERENCE_IMAGE_SIZE,
    VEHICLE_INSTANCE_MATCH_MINIMUM_COMPONENT_FRACTION,
    VEHICLE_INSTANCE_MATCH_MINIMUM_INSTANCE_FRACTION,
    VEHICLE_INSTANCE_MATCH_MINIMUM_INTERSECTION_PIXELS,
    VEHICLE_PARTITION_DISTANCE_MASK_SIZE,
)


class VehicleSegmentationModel(Protocol):
    """Minimal lazy-injected interface required from a vehicle model."""

    names: Mapping[int, str]

    def predict(self, image: np.ndarray, **kwargs: object) -> Sequence[Any]: ...


class VehicleModelProvider(Protocol):
    """Lazy, configuration-owned access to a reusable vehicle model."""

    def get_model(self) -> VehicleSegmentationModel: ...


class UltralyticsVehicleModelProvider:
    """Lazy owner of the accepted local YOLO segmentation model."""

    def __init__(
        self,
        weights_path: Path,
        requested_device: str | None = None,
    ) -> None:
        self.weights_path = weights_path.resolve()
        self.requested_device = requested_device
        self._model: VehicleSegmentationModel | None = None
        self._synchronized_model: VehicleSegmentationModel | None = None
        self._model_lock = Lock()

    def get_model(self) -> VehicleSegmentationModel:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from ultralytics import YOLO

                    self._model = YOLO(str(self.weights_path))
                    self._synchronized_model = _SynchronizedVehicleModel(
                        self._model,
                        requested_device=self.requested_device,
                    )
        assert self._synchronized_model is not None
        return self._synchronized_model


class _SynchronizedVehicleModel:
    """Serialize calls into one reusable Ultralytics model instance."""

    def __init__(
        self,
        model: VehicleSegmentationModel,
        *,
        requested_device: str | None = None,
    ) -> None:
        self._model = model
        self._requested_device = requested_device
        self._predict_lock = Lock()

    @property
    def names(self) -> Mapping[int, str]:
        return self._model.names

    def predict(self, image: np.ndarray, **kwargs: object) -> Sequence[Any]:
        if self._requested_device is not None:
            kwargs["device"] = (
                "0" if self._requested_device == "cuda"
                else self._requested_device
            )
        with self._predict_lock:
            return self._model.predict(image, **kwargs)


def rasterize_instance_polygon(
    points: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize one model polygon into a boolean image-space mask."""

    mask = Image.new("L", (shape[1], shape[0]), 0)
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon(
            [tuple(map(float, point)) for point in points],
            fill=255,
        )
    return np.asarray(mask) > 0


def infer_vehicle_instances(
    model: VehicleSegmentationModel,
    image: np.ndarray,
) -> VehicleInferenceResult:
    """Run one injected model call and retain only supported vehicle classes."""

    result = model.predict(
        image,
        device=os.environ.get(
            VEHICLE_INFERENCE_DEVICE_ENVIRONMENT_VARIABLE,
            VEHICLE_INFERENCE_DEFAULT_DEVICE,
        ),
        imgsz=VEHICLE_INFERENCE_IMAGE_SIZE,
        conf=VEHICLE_INFERENCE_CONFIDENCE_THRESHOLD,
        verbose=False,
    )[0]
    instances: list[VehicleInstance] = []
    if result.boxes is None:
        return VehicleInferenceResult(
            tuple(instances),
            VehicleInferenceDiagnostics(0, 0),
        )
    mask_polygons = result.masks.xy if result.masks is not None else []
    for index, box in enumerate(result.boxes):
        class_id = int(box.cls.item())
        name = str(model.names[class_id])
        if name not in VEHICLE_INSTANCE_CLASS_NAMES:
            continue
        xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
        mask = np.zeros(image.shape[:2], dtype=bool)
        if index < len(mask_polygons):
            mask = rasterize_instance_polygon(
                mask_polygons[index],
                image.shape[:2],
            )
        instances.append(VehicleInstance(
            detection_index=index,
            class_id=class_id,
            class_name=name,
            confidence=float(box.conf.item()),
            xyxy=xyxy,
            mask_pixels=int(mask.sum()),
            mask=mask,
        ))
    return VehicleInferenceResult(
        tuple(instances),
        VehicleInferenceDiagnostics(
            box_count=int(len(result.boxes)),
            mask_count=(
                0
                if result.masks is None
                else int(len(result.masks.xy))
            ),
        ),
    )


def match_vehicle_instances(
    component_mask: np.ndarray,
    instances: tuple[VehicleInstance, ...] | list[VehicleInstance],
) -> VehicleInstanceMatchEvidence:
    """Match vehicle instances to one triggered Mapillary car component."""

    accepted: list[VehicleInstance] = []
    audit: list[VehicleInstanceMatch] = []
    component_pixels = max(1, int(component_mask.sum()))
    for instance in instances:
        intersection = int((instance.mask & component_mask).sum())
        instance_fraction = intersection / max(1, int(instance.mask.sum()))
        component_fraction = intersection / component_pixels
        matched = (
            intersection
            >= VEHICLE_INSTANCE_MATCH_MINIMUM_INTERSECTION_PIXELS
            and (
                instance_fraction
                >= VEHICLE_INSTANCE_MATCH_MINIMUM_INSTANCE_FRACTION
                or component_fraction
                >= VEHICLE_INSTANCE_MATCH_MINIMUM_COMPONENT_FRACTION
            )
        )
        audit.append(VehicleInstanceMatch(
            instance=instance,
            mapillary_intersection_pixels=intersection,
            yolo_mask_overlap_fraction=instance_fraction,
            mapillary_component_overlap_fraction=component_fraction,
            matched=bool(matched),
        ))
        if matched:
            accepted.append(instance)
    return VehicleInstanceMatchEvidence(tuple(accepted), tuple(audit))


def partition_mapillary_component(
    component_mask: np.ndarray,
    instances: tuple[VehicleInstance, ...] | list[VehicleInstance],
) -> VehicleInstancePartition:
    """Assign every parent component pixel to its nearest instance seed."""

    distances = []
    for instance in instances:
        seed = instance.mask & component_mask
        outside_seed = (~seed).astype(np.uint8)
        distances.append(cv2.distanceTransform(
            outside_seed,
            cv2.DIST_L2,
            VEHICLE_PARTITION_DISTANCE_MASK_SIZE,
        ))
    winner = np.argmin(np.stack(distances, axis=0), axis=0)
    return VehicleInstancePartition(tuple(
        component_mask & (winner == index)
        for index in range(len(instances))
    ))
