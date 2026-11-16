"""Single-GPU adapter training, deterministic validation, and resumable checkpoints."""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import json
import logging
import math
import os
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Subset

from .config import TrainConfig
from .data import AudioTimeDataset, collate_audio, read_manifest, sha256_file
from .decomposer import PromptDecomposer
from .system import batch_objective, load_backbone


def data_fingerprint(config):
    paths = [
        Path(config.manifest_dir) / f"{split}.jsonl"
        for split in ("train", "validation")
    ]
    return {p.name: sha256_file(p) for p in paths}


def backbone_fingerprint(config):
    root = Path(config.model_dir)
    files = ("model_config.json", "download_manifest.json")
    if not (root / "download_manifest.json").exists():
        raise ValueError("Expected the verified model download_manifest.json")
    return {name: sha256_file(root / name) for name in files}


def preflight(config: TrainConfig):
    """CPU-only checks. Do not instantiate/load the complete backbone or touch CUDA."""
    root = Path(config.model_dir)
    raw = json.loads((root / "model_config.json").read_text())
    if raw["model"]["diffusion"]["diffusion_objective"] != "rectified_flow":
        raise ValueError("Expected rectified-flow model")
    if raw["sample_rate"] != config.sample_rate or raw["audio_channels"] != 2:
        raise ValueError("Sample rate/audio channels do not match the backbone")
    ae = raw["model"]["pretransform"]["config"]
    if ae["downsampling_ratio"] != config.downsampling_ratio:
        raise ValueError("Latent downsampling ratio mismatch")
    if raw["model"]["conditioning"]["cond_dim"] != config.adapter.text_dim:
        raise ValueError("PRISM output dimension does not match SA3 conditioning")
    if config.latent_alignment != 2:
        raise ValueError("This Medium Base preset expects an even latent length")
    verified = json.loads((root / "download_manifest.json").read_text())
    if verified["repo_id"] != "stabilityai/stable-audio-3-medium-base":
        raise ValueError("This preset expects the official Medium Base snapshot")
    for record in verified["files"]:
        path = root / record["path"]
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Model file size changed: {path}")
    train = AudioTimeDataset(Path(config.manifest_dir) / "train.jsonl", config)
    validation = AudioTimeDataset(
        Path(config.manifest_dir) / "validation.jsonl", config
    )
    test = read_manifest(Path(config.manifest_dir) / "test.jsonl")
    groups = [
        {r["audio_sha256"] for r in rows}
        for rows in (train.records, validation.records, test)
    ]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Audio content leakage across train/validation/test manifests")
    for row in train.records + validation.records + test:
        if not Path(row["audio_path"]).is_file():
            raise FileNotFoundError(row["audio_path"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        root / "t5gemma-b-b-ul2", local_files_only=True
    )
    maximum = 0
    rows = train.records + validation.records + test
    for start in range(0, len(rows), 256):
        lengths = tokenizer(
            [r["caption"] for r in rows[start : start + 256]],
            truncation=False,
            return_length=True,
        )["length"]
        maximum = max(maximum, max(lengths))
    if maximum > config.text_max_length:
        raise ValueError(
            f"Caption length {maximum} exceeds text_max_length={config.text_max_length}"
        )
    adapter = PromptDecomposer(config.adapter)
    if train.samples > raw["sample_size"]:
        raise ValueError("Padded training clip exceeds the backbone sample_size")
    example = train[0]
    return {
        "status": "CPU preflight passed; no training or GPU execution",
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "trainable_adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "max_caption_tokens": maximum,
        "text_max_length": config.text_max_length,
        "waveform_shape": list(example["audio"].shape),
        "latent_length": train.samples // config.downsampling_ratio,
        "data_fingerprint": data_fingerprint(config),
        "backbone_fingerprint": backbone_fingerprint(config),
        "config": config.to_dict(),
    }


def checkpoint_payload(adapter, optimizer, config, position, fingerprints):
    return {
        "format": "sa3-prism-v1",
        "adapter_config": asdict(config.adapter),
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "position": position,
        "fingerprints": fingerprints,
        "rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
        },
    }


def save_checkpoint(path, adapter, optimizer, config, position, fingerprints):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        checkpoint_payload(adapter, optimizer, config, position, fingerprints),
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(path, adapter, optimizer, config, fingerprints):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "sa3-prism-v1" or state["adapter_config"] != asdict(
        config.adapter
    ):
        raise ValueError("Incompatible PRISM checkpoint format/architecture")
    if state["fingerprints"] != fingerprints:
        raise ValueError("Backbone or training data changed since checkpoint")
    mutable = {
        "max_steps",
        "output_dir",
        "workers",
        "log_every",
        "checkpoint_every",
        "validate_every",
        "validation_samples",
    }
    for key, value in config.to_dict().items():
        if key not in mutable and state["config"][key] != value:
            raise ValueError(f"Cannot resume with changed setting: {key}")
    adapter.load_state_dict(state["adapter"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"]["torch"])
    random.setstate(state["rng"]["python"])
    if state["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state(state["rng"]["cuda"])
    return state["position"]


def validation_indices(records, count, seed):
    """Round robin across tasks so small validation runs cover all four tasks."""
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[row["task"]].append(index)
    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)
    chosen = []
    for offset in range(max(map(len, groups.values()))):
        for task in sorted(groups):
            if offset < len(groups[task]):
                chosen.append(groups[task][offset])
                if len(chosen) == min(count, len(records)):
                    return chosen
    return chosen


@contextmanager
def validation_rng(seed, device):
    devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


@torch.no_grad()
def validate(backbone, adapter, dataset, config, device):
    adapter.eval()
    indices = validation_indices(
        dataset.records, config.validation_samples, config.seed
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_audio,
        num_workers=0,
        generator=torch.Generator().manual_seed(config.seed),
    )
    totals, count = defaultdict(float), 0
    try:
        with validation_rng(config.seed + 1, device):
            for batch in loader:
                _, metrics = batch_objective(
                    backbone, adapter, batch, config, device, validation=True
                )
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Nonfinite validation metric")
                for key, value in metrics.items():
                    totals[key] += value * len(batch["records"])
                count += len(batch["records"])
    finally:
        adapter.train()
    return {key: value / count for key, value in totals.items()}


def train(config, resume=None):
    # Explicit execution only; --preflight in the CLI returns before reaching here.
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "This initial implementation supports one GPU; do not launch with torchrun"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for full Medium Base training; use --preflight for CPU checks"
        )
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU lacks bfloat16 support")
    device = torch.device("cuda", 0)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(output / "train.log"), logging.StreamHandler()],
    )
    # Run data/model checks again to fail before allocating the large backbone.
    report = preflight(config)
    (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    fingerprints = {
        "data": report["data_fingerprint"],
        "backbone": report["backbone_fingerprint"],
    }
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    backbone, _ = load_backbone(
        config.model_dir, device, config.precision, config.text_max_length
    )
    adapter = PromptDecomposer(config.adapter).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    position = {"epoch": 0, "next_batch": 0, "step": 0}
    if resume:
        position = load_checkpoint(resume, adapter, optimizer, config, fingerprints)
        logging.info("Resumed %s at %s", resume, position)
    train_data = AudioTimeDataset(Path(config.manifest_dir) / "train.jsonl", config)
    val_data = AudioTimeDataset(Path(config.manifest_dir) / "validation.jsonl", config)
    adapter.train()
    optimizer.zero_grad(set_to_none=True)
    while position["step"] < config.max_steps:
        epoch = position["epoch"]
        loader = DataLoader(
            train_data,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed + epoch),
            num_workers=config.workers,
            collate_fn=collate_audio,
            pin_memory=True,
        )
        if not 0 <= position["next_batch"] < len(loader):
            raise ValueError("Invalid resume batch position")
        first_batch = position["next_batch"]
        accumulated_metrics, accumulated_examples = defaultdict(float), 0
        for index, batch in enumerate(loader):
            if index < first_batch:
                continue
            group_start = index // config.accumulation * config.accumulation
            group_size = min(config.accumulation, len(loader) - group_start)
            loss, metrics = batch_objective(backbone, adapter, batch, config, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite training loss at epoch={epoch}, batch={index}"
                )
            (loss / group_size).backward()
            examples = len(batch["records"])
            for key, value in metrics.items():
                accumulated_metrics[key] += value * examples
            accumulated_examples += examples
            if (index + 1) % config.accumulation and index + 1 != len(loader):
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), config.grad_clip, error_if_nonfinite=True
            )
            next_step = position["step"] + 1
            lr = config.learning_rate * min(
                1.0, next_step / max(1, config.warmup_steps)
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            position = {
                "epoch": epoch + int(index + 1 == len(loader)),
                "next_batch": 0 if index + 1 == len(loader) else index + 1,
                "step": next_step,
            }
            metrics = {
                key: value / accumulated_examples
                for key, value in accumulated_metrics.items()
            }
            accumulated_metrics, accumulated_examples = defaultdict(float), 0
            if next_step % config.log_every == 0 or next_step == 1:
                event = {
                    "split": "train",
                    **position,
                    **metrics,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(event) + "\n")
                logging.info("%s", json.dumps(event))
            if next_step % config.validate_every == 0:
                event = {
                    "split": "validation",
                    "step": next_step,
                    **validate(backbone, adapter, val_data, config, device),
                }
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(event) + "\n")
                logging.info("%s", json.dumps(event))
            if (
                next_step % config.checkpoint_every == 0
                or next_step == config.max_steps
            ):
                checkpoint_path = output / f"checkpoint-step-{next_step:06d}.pt"
                save_checkpoint(
                    checkpoint_path, adapter, optimizer, config, position, fingerprints
                )
                (output / "latest_checkpoint.txt").write_text(
                    str(checkpoint_path.resolve()) + "\n"
                )
                logging.info("Saved adapter-only checkpoint %s", checkpoint_path)
            if next_step >= config.max_steps:
                break
    logging.info("Training finished at step %s", position["step"])
