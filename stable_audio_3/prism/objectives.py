from itertools import permutations

import torch
from torch.nn import functional as F


def rectified_flow_pair(clean, noise, timesteps):
    """SA3 uses t=1 for noise and t=0 for data: target is noise - clean."""
    if clean.shape != noise.shape or timesteps.shape != (clean.shape[0],):
        raise ValueError("Mismatched RF inputs")
    t = timesteps.float()[:, None, None]
    return (1 - t) * clean + t * noise, noise - clean


def masked_mse(prediction, target, valid):
    if valid.shape != (prediction.shape[0], prediction.shape[-1]):
        raise ValueError("Loss mask shape mismatch")
    if not valid.any(dim=1).all():
        raise ValueError("Loss mask must contain signal for every example")
    squared = (prediction.float() - target.float()).square().mean(1)
    return ((squared * valid).sum(-1) / valid.sum(-1)).mean()


def orthogonality_loss(components):
    """Public SD1.5 training code regularizes squared pairwise pooled cosine."""
    k = components.shape[1]
    if k == 1:
        return components.sum() * 0
    pooled = F.normalize(components.float().mean(2), dim=-1, eps=1e-8)
    pair = pooled @ pooled.transpose(1, 2)
    i, j = torch.triu_indices(k, k, offset=1, device=pair.device)
    return pair[:, i, j].square().mean()


def temporal_matching_loss(
    components,
    logits,
    targets,
    valid,
    event_features=None,
    temporal_weight=1.0,
    semantic_weight=0.0,
):
    """Permutation-invariant auxiliary supervision; annotations NEVER enter inference.

    Exact interval supervision is used only for ordering/timestamp. Scalar lists
    from duration/frequency do not specify event boundaries. Unused slots receive
    zero activity targets only when interval annotations are available.
    """
    zero = components.sum() * 0
    temporal_losses, semantic_losses = [], []
    if logits is None:
        return zero, zero
    k = components.shape[1]
    pooled = F.normalize(components.float().mean(2), dim=-1, eps=1e-8)
    for b, sample in enumerate(targets):
        events = sample["events"]
        n = len(events)
        if n > k:
            raise ValueError(f"{sample['uid']}: {n} events exceed {k} components")
        if not n:
            continue
        activity = logits.new_zeros((k, logits.shape[-1]), dtype=torch.float32)
        if sample["has_intervals"]:
            times = sample["latent_times"].to(logits.device)
            for e, event in enumerate(events):
                for start, end in event["intervals"]:
                    activity[e] = torch.maximum(
                        activity[e], ((times >= start) & (times < end)).float()
                    )
            # Matrix [predicted slot, target slot], including background slots.
            bce = F.binary_cross_entropy_with_logits(
                logits[b, :, None].expand(-1, k, -1).float(),
                activity[None].expand(k, -1, -1),
                reduction="none",
            )
            temporal_cost = (bce * valid[b]).sum(-1) / valid[b].sum().clamp_min(1)
        else:
            temporal_cost = logits.new_zeros(k, k, dtype=torch.float32)
        semantic_cost = logits.new_zeros(k, k, dtype=torch.float32)
        if event_features is not None and semantic_weight:
            semantic_cost[:, :n] = (
                1 - pooled[b] @ F.normalize(event_features[b][:n].float(), dim=-1).T
            )
        choices = torch.tensor(list(permutations(range(k))), device=logits.device)
        rows = torch.arange(k, device=logits.device)[None].expand_as(choices)
        joint = temporal_weight * temporal_cost + semantic_weight * semantic_cost * (
            k / n
        )
        best = joint.detach()[rows, choices].mean(1).argmin()
        assignment = choices[best]
        if sample["has_intervals"]:
            temporal_losses.append(temporal_cost[rows[0], assignment].mean())
        if event_features is not None and semantic_weight:
            # The auxiliary semantic loss is normalized by actual event count.
            semantic_losses.append(semantic_cost[rows[0], assignment].sum() / n)
    return (
        torch.stack(temporal_losses).mean() if temporal_losses else zero,
        torch.stack(semantic_losses).mean() if semantic_losses else zero,
    )
