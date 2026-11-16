"""Generate from a trained adapter, optionally paired with the frozen baseline."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import soundfile as sf
import torch

from stable_audio_3.prism.config import TrainConfig
from stable_audio_3.prism.data import read_manifest
from stable_audio_3.prism.decomposer import PromptDecomposer
from stable_audio_3.prism.system import generate, load_backbone
from stable_audio_3.prism.training import backbone_fingerprint


def main():
    parser = argparse.ArgumentParser(
        description="Local PRISM text-to-audio; evaluation annotations are never model inputs"
    )
    parser.add_argument(
        "--checkpoint",
        help="Trained PRISM adapter checkpoint; required except for baseline-only",
    )
    parser.add_argument(
        "--config",
        default="configs/prism/audiotime.json",
        help="Used for baseline-only without checkpoint",
    )
    parser.add_argument(
        "--model-dir",
        help="Override local backbone path, with fingerprint verification",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt")
    inputs.add_argument(
        "--eval-manifest", help="For example data/prism_audiotime/test.jsonl"
    )
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Frozen SA3 only, using the same Euler/CFG sampler",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Generate both baseline and PRISM with identical seeds",
    )
    args = parser.parse_args()
    if args.baseline and args.paired:
        parser.error("--baseline and --paired are mutually exclusive")
    if not args.baseline and not args.checkpoint:
        parser.error(
            "A trained --checkpoint is required; random adapters are never silently used"
        )
    if args.limit < 1:
        parser.error("--limit must be positive")
    state = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if state.get("format") != "sa3-prism-v1":
            raise ValueError("Unsupported adapter checkpoint")
        config = TrainConfig(**state["config"])
    else:
        config = TrainConfig.read(args.config)
    if args.model_dir:
        config.model_dir = args.model_dir
    if state and state["fingerprints"]["backbone"] != backbone_fingerprint(config):
        raise ValueError("Checkpoint was trained with a different backbone snapshot")
    if not torch.cuda.is_available():
        raise RuntimeError("Medium Base generation requires a CUDA GPU")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "generations.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"Use a new output directory: {manifest_path}")
    records = (
        read_manifest(args.eval_manifest)[: args.limit]
        if args.eval_manifest
        else [{"uid": "custom:0001", "caption": args.prompt, "duration": args.duration}]
    )
    backbone, _ = load_backbone(
        config.model_dir, "cuda", config.precision, config.text_max_length
    )
    adapter = PromptDecomposer(config.adapter).cuda().eval()
    if state:
        adapter.load_state_dict(state["adapter"], strict=True)
    modes = (
        ("baseline", "prism")
        if args.paired
        else ("baseline",)
        if args.baseline
        else ("prism",)
    )
    for index, row in enumerate(records):
        for mode in modes:
            seed = args.seed + index
            audio, trace = generate(
                backbone,
                adapter,
                row["caption"],
                row["duration"],
                config,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                seed=seed,
                baseline=mode == "baseline",
            )
            stem = row["uid"].replace(":", "_").replace("/", "_")
            target = output / mode / f"{stem}.wav"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            sf.write(target, audio[0].float().cpu().T.numpy(), config.sample_rate)
            result = {
                "uid": row["uid"],
                "mode": mode,
                "caption": row["caption"],
                "duration": row["duration"],
                "audio_path": str(target.resolve()),
                "seed": seed,
                "steps": args.steps,
                "cfg_scale": args.cfg_scale,
                "sampler": "Euler + vanilla CFG (identical in both modes)",
                "adapter_checkpoint": args.checkpoint,
                "diagnostics": trace,
            }
            with manifest_path.open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
