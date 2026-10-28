"""Frozen development-B ROI/coverage reconstruction; no model or labels."""

import io
from dataclasses import dataclass, asdict
import numpy as np


def read_pcd(data, *, return_index=False):
    """Read uncompressed PCD with explicit field/type/count validation.

    Optional provenance returns(points, original PCD row indices, header count).
    Default return and filtering remain unchanged for historical callers.
    """
    f = io.BytesIO(data)
    header = {}
    for _ in range(100):
        line = f.readline().decode("ascii").strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        header[parts[0].upper()] = parts[1:]
        if parts[0].upper() == "DATA":
            break
    fields = header["FIELDS"]
    if not all((v in fields for v in ("x", "y", "z"))):
        raise ValueError("PCD requires x, y, z fields")
    count = list(map(int, header.get("COUNT", ["1"] * len(fields))))
    size = list(map(int, header["SIZE"]))
    types = header["TYPE"]
    if not len(fields) == len(count) == len(size) == len(types) or any(
        (c != 1 for c in count)
    ):
        raise ValueError("Unsupported PCD field layout")
    n = int(
        header.get("POINTS", [str(int(header["WIDTH"][0]) * int(header["HEIGHT"][0]))])[
            0
        ]
    )
    if not 0 < n <= 5000000:
        raise ValueError("PCD point count out of range")
    mode = header["DATA"][0]
    if mode == "binary":
        codes = {"F": "f", "I": "i", "U": "u"}
        dtype = np.dtype(
            [(name, "<" + codes[t] + str(s)) for name, t, s in zip(fields, types, size)]
        )
        header_bytes = f.tell()
        payload = f.read()
        expected = n * dtype.itemsize
        if len(payload) != expected:
            padded = (
                header_bytes < 4096
                and len(payload) == expected + 4096 - header_bytes
                and (not any(payload[expected:]))
            )
            if not padded:
                raise ValueError("PCD payload length differs from header")
        a = np.frombuffer(payload, dtype=dtype, count=n)
        intensity = next(
            (v for v in ("reflectivity", "intensity") if v in fields), None
        )
        p = np.column_stack(
            [a["x"], a["y"], a["z"], a[intensity] if intensity else np.zeros(n)]
        )
    elif mode == "ascii":
        a = np.loadtxt(f, ndmin=2)
        if len(a) != n or a.shape[1] != len(fields):
            raise ValueError("PCD ASCII shape differs from header")
        ix = next(
            (fields.index(v) for v in ("reflectivity", "intensity") if v in fields),
            None,
        )
        p = np.column_stack(
            [a[:, fields.index(v)] for v in ("x", "y", "z")]
            + [a[:, ix] if ix is not None else np.zeros(n)]
        )
    else:
        raise ValueError(
            "Only binary or ASCII PCD is supported; convert binary_compressed first"
        )
    good = np.isfinite(p).all(1) & (np.linalg.norm(p[:, :3], axis=1) > 0.3)
    p = np.asarray(p[good], dtype=np.float32)
    if len(p) < 100:
        raise ValueError("Insufficient valid points")
    if return_index:
        return (p, np.flatnonzero(good).astype(np.int64), n)
    return p


@dataclass(frozen=True)
class InputContract:
    xy_half_extent_m: float = 20.0
    voxel_m: float = 0.05
    z_limits: tuple | None = None
    origin_exclusion_m: float = 0.3
    intensity_divisor: float = 255.0
    label_tolerance_m: float = 0.005

    def __post_init__(self):
        if (
            self.xy_half_extent_m <= 0
            or self.voxel_m <= 0
            or self.intensity_divisor <= 0
        ):
            raise ValueError("Positive ROI/voxel/intensity divisor required")
        if self.origin_exclusion_m != 0.3:
            raise ValueError(
                "Reader near-origin policy is fixed at0.3m;change explicitly before use"
            )
        if self.z_limits is not None and (not self.z_limits[0] < self.z_limits[1]):
            raise ValueError("Ordered height limits required")


def prepare_frame(
    raw, annotation=None, contract=InputContract(), *, build_voxel_groups=True
):
    """Labels never influence geometry/features or point selection."""
    points, pcd_index, origin_count = read_pcd(raw, return_index=True)
    roi = (np.abs(points[:, 0]) < contract.xy_half_extent_m) & (
        np.abs(points[:, 1]) < contract.xy_half_extent_m
    )
    if contract.z_limits is not None:
        roi &= (points[:, 2] > contract.z_limits[0]) & (
            points[:, 2] < contract.z_limits[1]
        )
    p = points[roi]
    if not len(p):
        raise ValueError("Empty ROI")
    feat = np.column_stack((p[:, :3], p[:, 3] / contract.intensity_divisor)).astype(
        np.float32
    )
    frame = dict(
        coord=p[:, :3].copy(),
        feat=feat,
        raw_pcd_index=pcd_index[roi],
        original_pcd_count=origin_count,
        parsed_points=len(points),
        roi_points=len(p),
        config=asdict(contract),
    )
    if build_voxel_groups:
        grid = np.floor(p[:, :3].astype(np.float64) / contract.voxel_m).astype(np.int64)
        grid -= grid.min(0)
        unique, first, inverse, count = np.unique(
            grid, axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        order = np.argsort(inverse, kind="stable")
        starts = np.cumsum(np.r_[0, count[:-1]])
        frame.update(
            grid_coord=grid,
            unique_grid_coord=unique,
            first=first,
            inverse=inverse,
            count=count,
            order=order,
            starts=starts,
        )
    if annotation is not None:
        raise ValueError("Annotations are not accepted by the inference input contract")
    return frame
