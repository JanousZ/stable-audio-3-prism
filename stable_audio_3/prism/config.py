from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class AdapterConfig:
    text_dim: int = 768
    width: int = 512
    heads: int = 8
    layers: int = 6
    components: int = 4
    tokens: int = 64
    temporal: bool = False
    gate_floor: float = 0.1

    def __post_init__(self):
        if (
            min(
                self.text_dim,
                self.width,
                self.heads,
                self.layers,
                self.components,
                self.tokens,
            )
            < 1
        ):
            raise ValueError("Adapter dimensions must be positive")
        if self.width % self.heads or self.width % 2:
            raise ValueError("width must be even and divisible by heads")
        if self.components > 6:
            raise ValueError(
                "At most six components are supported by exact temporal matching"
            )
        if not 0 <= self.gate_floor <= 1:
            raise ValueError("gate_floor must be in [0, 1]")


@dataclass
class TrainConfig:
    model_dir: str = "/mnt/disk1/models/stable-audio-3-medium-base"
    manifest_dir: str = "data/prism_audiotime"
    output_dir: str = "runs/prism_audiotime"
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    seed: int = 3467
    text_max_length: int = 512
    clip_seconds: float = 10.0
    headroom_seconds: float = 6.0
    sample_rate: int = 44100
    downsampling_ratio: int = 4096
    latent_alignment: int = 2
    batch_size: int = 1
    accumulation: int = 8
    workers: int = 2
    max_steps: int = 10000
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    precision: str = "bf16"
    component_chunk_size: int = 1
    gradient_checkpointing: bool = True
    orthogonality_weight: float = 0.1
    temporal_weight: float = 0.0
    semantic_weight: float = 0.0
    checkpoint_every: int = 500
    validate_every: int = 250
    validation_samples: int = 64
    log_every: int = 10

    def __post_init__(self):
        if isinstance(self.adapter, dict):
            self.adapter = AdapterConfig(**self.adapter)
        positive = (
            self.text_max_length,
            self.clip_seconds,
            self.sample_rate,
            self.downsampling_ratio,
            self.latent_alignment,
            self.batch_size,
            self.accumulation,
            self.max_steps,
            self.learning_rate,
            self.grad_clip,
            self.component_chunk_size,
            self.checkpoint_every,
            self.validate_every,
            self.validation_samples,
            self.log_every,
        )
        if (
            min(positive) <= 0
            or self.workers < 0
            or self.headroom_seconds < 0
            or self.warmup_steps < 0
        ):
            raise ValueError("Invalid nonpositive training setting")
        if (
            min(self.orthogonality_weight, self.temporal_weight, self.semantic_weight)
            < 0
        ):
            raise ValueError("Loss weights must be nonnegative")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("Use bf16 (recommended on A100) or fp32")
        if not self.adapter.temporal and (self.temporal_weight or self.semantic_weight):
            raise ValueError(
                "Temporal/semantic supervision requires adapter.temporal=true"
            )

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)
