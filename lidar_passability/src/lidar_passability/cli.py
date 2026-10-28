"""Explicit output file, never overwrite; library analysis itself performs no writes."""

import argparse
import json
from pathlib import Path
from .analyzer import LidarAnalyzer
from .contracts import AnalysisRequest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--request",
        type=Path,
        required=True,
        help="JSON AnalysisRequest; relative paths resolve beside this JSON",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New result JSON; parent must already exist",
    )
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Existing output is preserved")
    data = json.loads(args.request.read_text(encoding="utf8"))
    for key in ("pcd_path", "prediction_path", "report_path"):
        path = Path(data[key])
        data[key] = path if path.is_absolute() else args.request.resolve().parent / path
    result = LidarAnalyzer().analyze_prediction(AnalysisRequest(**data))
    with args.output.open("x", encoding="utf8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, allow_nan=False)
    print(
        json.dumps(
            dict(
                processing_status="completed",
                output=str(args.output),
                production_eligible=False,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
