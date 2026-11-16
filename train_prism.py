import argparse
import json
from pathlib import Path

from stable_audio_3.prism.config import TrainConfig
from stable_audio_3.prism.training import preflight, train


def main():
    parser = argparse.ArgumentParser(
        description="Train the PRISM adapter on a frozen SA3 Medium Base"
    )
    parser.add_argument("--config", default="configs/prism/audiotime.json")
    parser.add_argument(
        "--preflight",
        "--dry-run",
        action="store_true",
        help="CPU checks only; never loads full backbone or starts training",
    )
    parser.add_argument("--report", help="Optional JSON output for CPU preflight")
    parser.add_argument(
        "--resume", help="Adapter checkpoint including optimizer and RNG state"
    )
    parser.add_argument(
        "--max-steps", type=int, help="Explicitly override optimizer step budget"
    )
    args = parser.parse_args()
    config = TrainConfig.read(args.config)
    if args.max_steps is not None:
        config.max_steps = args.max_steps
        config.__post_init__()
    if args.preflight:
        report = preflight(config)
        if args.report:
            path = Path(args.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    else:
        train(config, args.resume)


if __name__ == "__main__":
    main()
