"""Query-based bidirectional-encoder PRISM branch (paper sections 3.3, A.1).

Independent implementation using existing PyTorch dependencies. Unlike the public
SD1.5 snapshot, query normalization is not time-adaptive, as specified in A.1.
"""

import math

import torch
from torch import nn

from .config import AdapterConfig


class AdaptiveNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Linear(width, 2 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, time):
        shift, scale = self.modulation(torch.nn.functional.silu(time)).chunk(2, -1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class QueryBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(width, eps=1e-6)
        self.text_norm = AdaptiveNorm(width)
        self.ff_norm = AdaptiveNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width)
        )

    def forward(self, query, text, valid_text, time):
        q = self.query_norm(query)
        kv = torch.cat((q, self.text_norm(text, time)), dim=1)
        valid = torch.cat(
            (torch.ones(q.shape[:2], device=q.device, dtype=torch.bool), valid_text),
            dim=1,
        )
        query = (
            query
            + self.attention(q, kv, kv, key_padding_mask=~valid, need_weights=False)[0]
        )
        return query + self.ff(self.ff_norm(query, time))


class PromptDecomposer(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        w = config.width
        self.queries = nn.Parameter(
            torch.randn(config.components, config.tokens, w) / math.sqrt(w)
        )
        self.input_projection = nn.Linear(config.text_dim, w)
        self.time_mlp = nn.Sequential(
            nn.Linear(w, w * 4), nn.SiLU(), nn.Linear(w * 4, w)
        )
        self.blocks = nn.ModuleList(
            QueryBlock(w, config.heads) for _ in range(config.layers)
        )
        self.output_projection = nn.Sequential(
            nn.Linear(w, config.text_dim), nn.LayerNorm(config.text_dim)
        )
        if config.temporal:
            # This is an audio-specific extension, not a claim about the PRISM paper.
            self.gate_slot = nn.Linear(config.text_dim, w)
            self.gate_text = nn.Linear(config.text_dim, w)
            self.gate_position = nn.Sequential(
                nn.Linear(6, w), nn.SiLU(), nn.Linear(w, w)
            )
            self.gate_output = nn.Sequential(nn.SiLU(), nn.Linear(w, 1))

    def forward(self, text, text_mask, timesteps, positions=None, durations=None):
        if (
            text.ndim != 3
            or text.shape[:2] != text_mask.shape
            or text.shape[-1] != self.config.text_dim
        ):
            raise ValueError("Expected text [B,S,D] and text_mask [B,S]")
        b, _, _ = text.shape
        if timesteps.shape != (b,) or not text_mask.bool().any(dim=1).all():
            raise ValueError("Each caption needs a valid token and one timestep")
        # RF t is kept in float32; the 1000 scale matches PRISM's denoising-time input.
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(self.config.width // 2, device=text.device).float()
            / (self.config.width // 2)
        )
        phase = timesteps.float()[:, None] * 1000 * frequencies[None]
        time = self.time_mlp(
            torch.cat((phase.cos(), phase.sin()), -1).to(self.queries.dtype)
        )
        k = self.config.components
        queries = (
            self.queries[None]
            .expand(b, -1, -1, -1)
            .reshape(b * k, self.config.tokens, -1)
        )
        features = self.input_projection(text.to(self.queries.dtype)).repeat_interleave(
            k, 0
        )
        mask = text_mask.bool().repeat_interleave(k, 0)
        time = time.repeat_interleave(k, 0)
        for block in self.blocks:
            queries = block(queries, features, mask, time)
        components = self.output_projection(queries).reshape(
            b, k, self.config.tokens, self.config.text_dim
        )
        logits = None
        if self.config.temporal:
            if positions is None or durations is None or durations.shape != (b,):
                raise ValueError(
                    "Temporal mode needs normalized latent positions and durations"
                )
            if positions.ndim == 1:
                positions = positions[None].expand(b, -1)
            if positions.shape[0] != b:
                raise ValueError("Temporal positions have the wrong batch size")
            p = positions.to(components.dtype)
            seconds = durations[:, None].expand_as(p).to(p.dtype)
            pos = torch.stack(
                (
                    p,
                    p.square(),
                    (p * math.pi).sin(),
                    (p * math.pi).cos(),
                    p * seconds / 10,
                    seconds / 380,
                ),
                -1,
            )
            pooled_text = (text * text_mask[..., None]).sum(1) / text_mask.sum(
                1, keepdim=True
            )
            hidden = (
                self.gate_slot(components.mean(2))[:, :, None]
                + self.gate_text(pooled_text.to(components.dtype))[:, None, None]
                + self.gate_position(pos)[:, None]
            )
            logits = self.gate_output(hidden).squeeze(-1)
        return {"components": components, "gate_logits": logits}

    def composition_weights(self, logits, batch, length, device):
        k = self.config.components
        if logits is None:
            return torch.full((batch, k, length), 1 / k, device=device)
        activity = logits.float().sigmoid().clamp_min(1e-6)
        weights = activity / activity.sum(1, keepdim=True).clamp_min(1e-8)
        return self.config.gate_floor / k + (1 - self.config.gate_floor) * weights
