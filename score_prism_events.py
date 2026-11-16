import argparse
import json
from pathlib import Path

from stable_audio_3.prism.data import read_manifest
from stable_audio_3.prism.evaluation import score_detections


def main():
    parser = argparse.ArgumentParser(
        description="Score externally detected audio events against an AudioTime manifest"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--detections",
        required=True,
        help='JSONL: {"uid": ..., "events": [{"name": ..., "start": 0, "end": 1}]}',
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.iou_threshold <= 1:
        parser.error("--iou-threshold must be in (0, 1]")
    detections = [
        json.loads(line)
        for line in Path(args.detections).read_text().splitlines()
        if line.strip()
    ]
    result = score_detections(
        read_manifest(args.manifest), detections, args.iou_threshold
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
