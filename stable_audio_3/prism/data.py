"""AudioTime's four annotation schemas, leak-aware splits, and whole-clip loading."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import torchaudio

TASKS = ("duration", "frequency", "ordering", "timestamp")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_events(annotation, task, duration):
    events = []
    for name, values in annotation["event"].items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Invalid event values: {name}")
        intervals = None
        if task in ("ordering", "timestamp"):
            intervals = []
            for pair in values:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError(f"Expected start/end interval for {name}")
                start, end = map(float, pair)
                if not (
                    math.isfinite(start)
                    and math.isfinite(end)
                    and 0 <= start < end <= duration + 0.02
                ):
                    raise ValueError(
                        f"Out-of-range event interval {pair}, duration={duration}"
                    )
                intervals.append([start, min(end, duration)])
        else:
            if any(
                not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                for v in values
            ):
                raise ValueError(f"Expected scalar duration list for {task}: {name}")
        events.append({"name": name, "intervals": intervals, "values": values})
    if not events:
        raise ValueError("Sample has no events")
    return events


def inspect_record(job):
    folder, audio_id, annotation, split, task = job
    audio = folder / "audio" / f"{audio_id}.wav"
    info = sf.info(audio)
    if not info.frames or info.channels not in (1, 2) or info.samplerate <= 0:
        raise ValueError(f"Invalid audio: {audio}")
    caption = annotation.get("caption", "")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(f"Missing caption: {audio}")
    return {
        "uid": f"{folder.name}:{audio_id}",
        "official_split": split,
        "task": task,
        "audio_path": str(audio.resolve()),
        "caption": caption,
        "duration": info.frames / info.samplerate,
        "frames": info.frames,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "audio_sha256": sha256_file(audio),
        "has_intervals": task in ("ordering", "timestamp"),
        "events": parse_events(annotation, task, info.frames / info.samplerate),
    }


def prepare_manifests(root, output, validation_fraction=0.05, seed=3467, workers=4):
    root, output = Path(root), Path(output)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    jobs = []
    for split, count in (("train", 5000), ("test", 500)):
        for task in TASKS:
            folder = root / f"{split}{count}_{task}"
            annotations = json.loads((folder / f"{task}_captions.json").read_text())
            if len(annotations) != count:
                raise ValueError(
                    f"{folder}: expected {count} annotations, got {len(annotations)}"
                )
            for audio_id, annotation in sorted(annotations.items()):
                jobs.append((folder, audio_id, annotation, split, task))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(inspect_record, jobs))
    test = [r for r in records if r["official_split"] == "test"]
    test_hashes = {r["audio_sha256"] for r in test}
    train_source = [r for r in records if r["official_split"] == "train"]
    # Keep the official test set intact; exclude identical train audio if needed.
    excluded = [r for r in train_source if r["audio_sha256"] in test_hashes]
    eligible = [r for r in train_source if r["audio_sha256"] not in test_hashes]
    train, validation = [], []
    for record in eligible:
        group = record["audio_sha256"]
        value = (
            int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
        )
        (validation if value < validation_fraction else train).append(record)
    if not train or not validation:
        raise ValueError("Empty train/validation split")
    output.mkdir(parents=True, exist_ok=True)
    partitions = {"train": train, "validation": validation, "test": test}
    for split, rows in partitions.items():
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    summary = {
        "dataset_root": str(root.resolve()),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "official_counts": dict(Counter(r["official_split"] for r in records)),
        "excluded_train_test_duplicates": [r["uid"] for r in excluded],
        "grouping": "SHA256 of complete WAV files; duplicate groups never cross splits",
        "partitions": {
            name: {
                "count": len(rows),
                "tasks": dict(Counter(r["task"] for r in rows)),
                "unique_audio": len({r["audio_sha256"] for r in rows}),
            }
            for name, rows in partitions.items()
        },
        "audio_duration_range": [
            min(r["duration"] for r in records),
            max(r["duration"] for r in records),
        ],
        "event_count_histogram": dict(Counter(len(r["events"]) for r in records)),
        "sample_rates": dict(Counter(r["sample_rate"] for r in records)),
        "interval_supervision_tasks": ["ordering", "timestamp"],
        "scalar_annotation_tasks": ["duration", "frequency"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def read_manifest(path):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not rows or len({r["uid"] for r in rows}) != len(rows):
        raise ValueError(f"Empty manifest or duplicate sample IDs: {path}")
    return rows


def aligned_samples(seconds, rate, ratio, alignment=2):
    unit = ratio * alignment
    return math.ceil(seconds * rate / unit) * unit


class AudioTimeDataset(Dataset):
    def __init__(self, manifest, config):
        self.records = read_manifest(manifest)
        self.config = config
        self.samples = aligned_samples(
            config.clip_seconds + config.headroom_seconds,
            config.sample_rate,
            config.downsampling_ratio,
            config.latent_alignment,
        )
        self.resamplers = {}
        for row in self.records:
            if row["duration"] > config.clip_seconds + 1 / row["sample_rate"]:
                raise ValueError(
                    f"{row['uid']}: duration exceeds clip_seconds; refusing temporal crop"
                )
            if (
                len(row["events"]) > config.adapter.components
                and config.adapter.temporal
            ):
                raise ValueError(
                    f"{row['uid']}: increase adapter.components for temporal matching"
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        audio, rate = sf.read(row["audio_path"], dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio.T.copy())
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        if rate != self.config.sample_rate:
            if rate not in self.resamplers:
                self.resamplers[rate] = torchaudio.transforms.Resample(
                    rate, self.config.sample_rate
                )
            audio = self.resamplers[rate](audio)
        if audio.shape[-1] > self.samples or not torch.isfinite(audio).all():
            raise ValueError(f"Invalid or overlong waveform: {row['uid']}")
        audio = F.pad(audio, (0, self.samples - audio.shape[-1]))
        return {"audio": audio, "record": row}


def collate_audio(examples):
    return {
        "audio": torch.stack([e["audio"] for e in examples]),
        "records": [e["record"] for e in examples],
    }


def latent_geometry(records, length, rate, ratio, headroom, device):
    # Left edges represent latent cells. This covers the last, partial cell too.
    absolute = torch.arange(length, device=device).float() * ratio / rate
    durations = torch.tensor(
        [r["duration"] for r in records], device=device, dtype=torch.float32
    )
    signal = absolute[None] < durations[:, None]
    attention = absolute[None] < (durations[:, None] + headroom)
    positions = absolute[None] / durations[:, None]
    targets = [{**r, "latent_times": absolute} for r in records]
    return durations, signal, attention, positions, targets
