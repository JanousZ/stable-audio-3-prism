import argparse
import json

from stable_audio_3.prism.data import prepare_manifests


def main():
    parser = argparse.ArgumentParser(
        description="CPU-only AudioTime audit and PRISM manifests"
    )
    parser.add_argument("--root", default="/mnt/disk1/datasets/AudioTime")
    parser.add_argument("--output", default="data/prism_audiotime")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3467)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    print(
        json.dumps(
            prepare_manifests(
                args.root,
                args.output,
                args.validation_fraction,
                args.seed,
                args.workers,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
