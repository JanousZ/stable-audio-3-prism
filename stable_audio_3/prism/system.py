"""Frozen SA3 backbone integration; gradients flow THROUGH the frozen DiT."""

from contextlib import nullcontext
import copy
import json
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from .data import aligned_samples, latent_geometry
from .objectives import (
    masked_mse,
    orthogonality_loss,
    rectified_flow_pair,
    temporal_matching_loss,
)


def autocast_context(device, precision):
    if precision == "bf16" and torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_backbone(model_dir, device, precision="bf16", text_max_length=512):
    from stable_audio_3.loading_utils import load_diffusion_cond

    root = Path(model_dir)
    config = json.loads((root / "model_config.json").read_text())
    if config["model"]["diffusion"]["diffusion_objective"] != "rectified_flow":
        raise ValueError("PRISM audio currently supports rectified-flow backbones only")
    effective = copy.deepcopy(config)
    native_text_length = None
    for c in effective["model"]["conditioning"]["configs"]:
        if c["type"] == "t5gemma":
            native_text_length = c["config"].get("max_length", 256)
            c["config"]["model_path"] = str((root / "t5gemma-b-b-ul2").resolve())
            c["config"]["max_length"] = text_max_length
            c["config"].pop("subfolder", None)
    if native_text_length is None:
        raise ValueError("Expected the SA3 T5Gemma text conditioner")
    model = load_diffusion_cond(
        effective, str(root / "model.safetensors"), device="cpu", model_half=False
    )
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    # Upstream stores the frozen text encoder in __dict__, outside nn.Module children.
    prompt = model.conditioner.conditioners["prompt"]
    prompt.model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    prompt._device_initialized = False
    model.prism_native_text_length = native_text_length
    return model, effective


@torch.no_grad()
def encode_conditioning(backbone, records, device, native=False):
    prompt = backbone.conditioner.conditioners["prompt"]
    previous = prompt.max_length
    if native:
        prompt.max_length = backbone.prism_native_text_length
    try:
        if not native:
            # Do not silently discard events beyond the configured encoder window.
            lengths = prompt.tokenizer(
                [r["caption"] for r in records], truncation=False, return_length=True
            )["length"]
            if max(lengths) > prompt.max_length:
                raise ValueError(
                    f"Caption exceeds text_max_length={prompt.max_length}; increase it explicitly"
                )
        return backbone.conditioner(
            [{"prompt": r["caption"], "seconds_total": r["duration"]} for r in records],
            device,
        )
    finally:
        prompt.max_length = previous


@torch.no_grad()
def encode_event_features(backbone, records, device):
    names = [f"The sound of {event['name']}." for r in records for event in r["events"]]
    features, mask = backbone.conditioner.conditioners["prompt"](names, device)
    pooled = (features.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    return list(pooled.split([len(r["events"]) for r in records]))


def generation_conditioning(conditioning, x):
    # Full text-to-audio generation: no known waveform and no retained inpaint area.
    return {
        **conditioning,
        "inpaint_mask": [x.new_zeros(x.shape[0], 1, x.shape[-1])],
        "inpaint_masked_input": [torch.zeros_like(x)],
    }


def base_velocity(backbone, x, t, conditioning, attention):
    return backbone(
        x,
        t.float(),
        cond=generation_conditioning(conditioning, x),
        padding_mask=attention,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
    )


class PrismPredictor:
    """Keep the 2B+ frozen parameters outside adapter/DDP checkpoint state."""

    def __init__(self, backbone, adapter, chunk_size=1, gradient_checkpointing=True):
        self.backbone = backbone
        self.adapter = adapter
        self.chunk_size = chunk_size
        self.gradient_checkpointing = gradient_checkpointing

    def __call__(self, x, t, conditioning, attention, positions, durations):
        decomposition = self.adapter(
            conditioning["prompt"][0],
            conditioning["prompt"][1],
            t.float(),
            positions,
            durations,
        )
        components = decomposition["components"]
        b, k, length, dim = components.shape
        flat_x = x[:, None].expand(-1, k, -1, -1).reshape(b * k, *x.shape[1:])
        flat_t = t.float().repeat_interleave(k)
        flat_text = components.reshape(b * k, length, dim)
        flat_attention = attention.repeat_interleave(k, 0)
        seconds = [v.repeat_interleave(k, 0) for v in conditioning["seconds_total"]]
        predictions = []
        for start in range(0, b * k, self.chunk_size):
            stop = min(start + self.chunk_size, b * k)
            part_seconds = [v[start:stop] for v in seconds]
            part_attention = flat_attention[start:stop]

            # Bind loop-specific conditioning before deferred checkpoint recomputation.
            def call(text, noisy, time, seconds=part_seconds, padding=part_attention):
                cond = {
                    "prompt": [
                        text,
                        torch.ones(
                            text.shape[:2], device=text.device, dtype=torch.bool
                        ),
                    ],
                    "seconds_total": seconds,
                }
                return base_velocity(self.backbone, noisy, time, cond, padding)

            args = (flat_text[start:stop], flat_x[start:stop], flat_t[start:stop])
            if (
                self.gradient_checkpointing
                and torch.is_grad_enabled()
                and flat_text.requires_grad
            ):
                predictions.append(checkpoint(call, *args, use_reentrant=False))
            else:
                predictions.append(call(*args))
        outputs = torch.cat(predictions).reshape(b, k, *x.shape[1:])
        weights = self.adapter.composition_weights(
            decomposition["gate_logits"], b, x.shape[-1], x.device
        )
        velocity = (outputs.float() * weights[:, :, None]).sum(1)
        return velocity, decomposition


def batch_objective(backbone, adapter, batch, config, device, *, validation=False):
    records = batch["records"]
    with torch.no_grad(), autocast_context(device, config.precision):
        audio = batch["audio"].to(device)
        clean = backbone.pretransform.encode(audio)
        conditioning = encode_conditioning(backbone, records, device)
        event_features = (
            encode_event_features(backbone, records, device)
            if config.adapter.temporal and config.semantic_weight
            else None
        )
    durations, signal, attention, positions, targets = latent_geometry(
        records,
        clean.shape[-1],
        config.sample_rate,
        config.downsampling_ratio,
        config.headroom_seconds,
        device,
    )
    # Match SA3's training timestep sampler, followed by its length-dependent shift.
    from stable_audio_3.inference.sampling import truncated_logistic_normal_rescaled

    t = 1 - truncated_logistic_normal_rescaled((len(records),)).to(device)
    if backbone.dist_shift is not None:
        t = backbone.dist_shift.shift(t, signal.sum(-1))
    noise = torch.randn_like(clean)
    noisy, target = rectified_flow_pair(clean, noise, t)
    predictor = PrismPredictor(
        backbone,
        adapter,
        config.component_chunk_size,
        config.gradient_checkpointing and not validation,
    )
    with autocast_context(device, config.precision):
        prediction, decomposition = predictor(
            noisy, t, conditioning, attention, positions, durations
        )
    flow = masked_mse(prediction, target, signal)
    ortho = orthogonality_loss(decomposition["components"])
    temporal, semantic = temporal_matching_loss(
        decomposition["components"],
        decomposition["gate_logits"],
        targets,
        signal,
        event_features,
        config.temporal_weight,
        config.semantic_weight,
    )
    loss = (
        flow
        + config.orthogonality_weight * ortho
        + config.temporal_weight * temporal
        + config.semantic_weight * semantic
    )
    metrics = {
        "loss": float(loss.detach()),
        "flow_mse": float(flow.detach()),
        "orthogonality": float(ortho.detach()),
        "temporal_bce": float(temporal.detach()),
        "semantic_cosine_loss": float(semantic.detach()),
    }
    if validation:
        with torch.no_grad(), autocast_context(device, config.precision):
            original_cond = encode_conditioning(backbone, records, device, native=True)
            original = base_velocity(backbone, noisy, t, original_cond, attention)
        metrics["baseline_flow_mse"] = float(masked_mse(original, target, signal))
    return loss, metrics


@torch.inference_mode()
def generate(
    backbone,
    adapter,
    caption,
    duration,
    config,
    *,
    steps=50,
    cfg_scale=7.0,
    seed=42,
    baseline=False,
):
    """One shared RF trajectory; component weights come from TEXT, not annotations."""
    if not 0 < duration <= 380 or steps < 1 or cfg_scale < 0:
        raise ValueError("Invalid duration, steps, or guidance")
    from stable_audio_3.inference.sampling import build_schedule, sample_discrete_euler

    device = next(backbone.model.parameters()).device
    dtype = next(backbone.model.parameters()).dtype
    samples = aligned_samples(
        duration + config.headroom_seconds,
        config.sample_rate,
        config.downsampling_ratio,
        config.latent_alignment,
    )
    if samples > 16777216:
        raise ValueError("Duration plus headroom exceeds Medium's maximum sample size")
    records = [{"caption": caption, "duration": duration, "events": []}]
    conditional = encode_conditioning(backbone, records, device, native=baseline)
    unconditional = encode_conditioning(
        backbone, [{**records[0], "caption": ""}], device, native=True
    )
    noise = torch.randn(
        (1, backbone.io_channels, samples // config.downsampling_ratio),
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
        dtype=dtype,
    )
    durations, _, attention, positions, _ = latent_geometry(
        records,
        noise.shape[-1],
        config.sample_rate,
        config.downsampling_ratio,
        config.headroom_seconds,
        device,
    )
    schedule = build_schedule(
        steps,
        dist_shift=backbone.sampling_dist_shift,
        effective_seq_len=math_ceil_tokens(duration, config),
        device=device,
    )
    predictor = PrismPredictor(backbone, adapter, config.component_chunk_size, False)
    trace = {}

    def velocity(x, t):
        with autocast_context(device, config.precision):
            if baseline:
                conditional_v = base_velocity(
                    backbone, x, t, conditional, attention
                ).float()
            else:
                conditional_v, decomposition = predictor(
                    x, t, conditional, attention, positions, durations
                )
                if decomposition["gate_logits"] is not None:
                    trace["last_gate_activity"] = (
                        decomposition["gate_logits"].float().sigmoid().cpu().tolist()
                    )
            if cfg_scale == 1:
                return conditional_v
            uncond_v = base_velocity(backbone, x, t, unconditional, attention).float()
            return uncond_v + cfg_scale * (conditional_v - uncond_v)

    latents = sample_discrete_euler(velocity, noise, schedule)
    with autocast_context(device, config.precision):
        audio = backbone.pretransform.decode(latents.to(dtype), chunked=True)
    audio = audio.float()[..., : round(duration * config.sample_rate)]
    if not torch.isfinite(audio).all():
        raise FloatingPointError("Nonfinite generated audio")
    return audio.clamp(-1, 1), trace


def math_ceil_tokens(duration, config):
    import math

    return math.ceil(int(duration * config.sample_rate) / config.downsampling_ratio)
