"""Binary PCD raw row reader; no filtering."""

from pathlib import Path
import io
import numpy as np


def read_all_raw(path):
    """Independently read every binary PCD row; no near-origin or ROI removal."""
    data = Path(path).read_bytes()
    f = io.BytesIO(data)
    header = {}
    for _ in range(80):
        line = f.readline().decode("ascii").strip()
        if not line or line.startswith("#"):
            continue
        key, *values = line.split()
        header[key] = values
        if key == "DATA":
            break
    if header["DATA"] != ["binary"]:
        raise ValueError("This bounded runner expects the selected binary PCD")
    fields = header["FIELDS"]
    sizes = list(map(int, header["SIZE"]))
    types = header["TYPE"]
    counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
    if len(set(fields)) != len(fields) or any((c != 1 for c in counts)):
        raise ValueError("Unsupported field layout")
    dtype = np.dtype(
        [
            (n, "<" + {"F": "f", "I": "i", "U": "u"}[t] + str(s))
            for n, t, s in zip(fields, types, sizes)
        ]
    )
    n = int(header["POINTS"][0])
    payload = f.read()
    if not 0 < n <= 150000 or len(payload) != n * dtype.itemsize:
        raise ValueError("Raw size outside one-scene plan")
    rows = np.frombuffer(payload, dtype=dtype, count=n)
    return np.column_stack([rows[a] for a in ("x", "y", "z")]).astype(np.float32)
