"""CPU-only contract tests. No pretrained weights, downloads, or GPU execution."""

from dataclasses import replace
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import soundfile as sf
import torch
from torch import nn

from stable_audio_3.prism.config import AdapterConfig, TrainConfig
from stable_audio_3.prism.data import AudioTimeDataset, latent_geometry, parse_events
from stable_audio_3.prism.decomposer import PromptDecomposer
from stable_audio_3.prism.evaluation import matched_intervals, score_detections
from stable_audio_3.prism.objectives import (
    masked_mse,
    orthogonality_loss,
    rectified_flow_pair,
    temporal_matching_loss,
)
from stable_audio_3.prism.system import PrismPredictor, generate
from stable_audio_3.prism.training import (
    load_checkpoint,
    save_checkpoint,
    validation_indices,
    validation_rng,
)

torch.set_num_threads(2)


def adapter_config(**kwargs):
    return AdapterConfig(
        text_dim=16, width=16, heads=2, layers=2, tokens=3, components=3, **kwargs
    )


def conditioning(batch=2):
    return {
        "prompt": [torch.randn(batch, 5, 16), torch.ones(batch, 5, dtype=torch.bool)],
        "seconds_total": [torch.randn(batch, 1, 16), torch.ones(batch, 1)],
    }


class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(16, 2, bias=False).requires_grad_(False)
        self.io_channels = 2
        self.sampling_dist_shift = None
        self.seen = []
        self.pretransform = type(
            "FakeCodec",
            (),
            {"decode": lambda _, x, chunked=True: x.repeat_interleave(2, -1)},
        )()

    def forward(self, x, t, cond, padding_mask, cfg_scale, cfg_dropout_prob):
        self.seen.append((x.detach().clone(), t.detach().clone(), cond))
        assert cfg_scale == 1 and cfg_dropout_prob == 0
        assert not cond["inpaint_mask"][0].any()
        assert not cond["inpaint_masked_input"][0].any()
        return x * 0.2 + self.model(cond["prompt"][0].mean(1)).unsqueeze(-1)


class DecompositionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_shapes_mask_and_time(self):
        module = PromptDecomposer(adapter_config())
        text = torch.randn(2, 5, 16)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
        output = module(text, mask, torch.tensor([0.2, 0.8]))
        changed = text.masked_fill(~mask[..., None], 90000)
        actual = module(changed, mask, torch.tensor([0.2, 0.8]))
        self.assertEqual(output["components"].shape, (2, 3, 3, 16))
        torch.testing.assert_close(output["components"], actual["components"])
        self.assertIsNone(output["gate_logits"])

    def test_reject_all_padding(self):
        with self.assertRaises(ValueError):
            PromptDecomposer(adapter_config())(
                torch.zeros(1, 2, 16), torch.zeros(1, 2), torch.zeros(1)
            )

    def test_components_are_distinct(self):
        m = PromptDecomposer(adapter_config())
        out = m(torch.randn(1, 4, 16), torch.ones(1, 4), torch.ones(1) * 0.5)[
            "components"
        ]
        self.assertGreater(float((out[:, 0] - out[:, 1]).abs().sum()), 0)

    def test_query_norm_is_not_time_adaptive(self):
        self.assertTrue(
            all(
                isinstance(block.query_norm, nn.LayerNorm)
                for block in PromptDecomposer(adapter_config()).blocks
            )
        )

    def test_temporal_normalization_and_overlap(self):
        m = PromptDecomposer(adapter_config(temporal=True))
        out = m(
            torch.randn(2, 4, 16),
            torch.ones(2, 4),
            torch.tensor([0.4, 0.6]),
            torch.linspace(0, 1, 9),
            torch.tensor([10.0, 8.0]),
        )
        weights = m.composition_weights(out["gate_logits"], 2, 9, "cpu")
        torch.testing.assert_close(weights.sum(1), torch.ones(2, 9))
        self.assertEqual(out["gate_logits"].shape, (2, 3, 9))
        self.assertTrue(torch.isfinite(weights).all())
        saturated = m.composition_weights(torch.full((2, 3, 9), -1000.0), 2, 9, "cpu")
        torch.testing.assert_close(saturated.sum(1), torch.ones(2, 9))


class FlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_rf_endpoints_and_direction(self):
        data, noise = torch.randn(2, 2, 7), torch.randn(2, 2, 7)
        x, target = rectified_flow_pair(data, noise, torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(x[0], data[0])
        torch.testing.assert_close(x[1], noise[1])
        torch.testing.assert_close(noise - target, data)

    def test_masked_padding_has_no_loss(self):
        prediction = torch.tensor([[[2.0, 900.0]]])
        self.assertEqual(
            float(
                masked_mse(
                    prediction,
                    torch.zeros_like(prediction),
                    torch.tensor([[True, False]]),
                )
            ),
            4,
        )

    def test_shared_noise_component_order_and_frozen_gradients(self):
        backbone = MockBackbone()
        module = PromptDecomposer(adapter_config())
        cond = conditioning()
        x, t = torch.randn(2, 2, 7), torch.tensor([0.2, 0.7])
        prediction, _ = PrismPredictor(backbone, module, 1, True)(
            x,
            t,
            cond,
            torch.ones(2, 7, dtype=torch.bool),
            torch.linspace(0, 1, 7),
            torch.ones(2),
        )
        # B-major flattening must give three copies of each example, not crossed captions.
        self.assertEqual(len(backbone.seen), 6)
        for i, (sample, time, _) in enumerate(backbone.seen):
            torch.testing.assert_close(sample, x[i // 3 : i // 3 + 1])
            torch.testing.assert_close(time, t[i // 3 : i // 3 + 1])
        prediction.square().mean().backward()
        self.assertGreater(float(module.queries.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))

    def test_chunked_matches_batched_forward_and_backward(self):
        backbone = MockBackbone()
        one = PromptDecomposer(adapter_config())
        two = copy.deepcopy(one)
        x, t, cond = torch.randn(2, 2, 7), torch.tensor([0.2, 0.7]), conditioning()
        args = (
            x,
            t,
            cond,
            torch.ones(2, 7, dtype=torch.bool),
            torch.linspace(0, 1, 7),
            torch.ones(2),
        )
        a, _ = PrismPredictor(backbone, one, 1, True)(*args)
        b, _ = PrismPredictor(backbone, two, 6, False)(*args)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        a.square().sum().backward()
        b.square().sum().backward()
        for p, q in zip(one.parameters(), two.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)

    def test_actual_sa3_dit_backward_on_cpu(self):
        from stable_audio_3.models.diffusion import (
            ConditionedDiffusionModelWrapper,
            DiTWrapper,
        )
        import stable_audio_3.models.transformer as transformer

        dit = DiTWrapper(
            "rectified_flow",
            io_channels=2,
            embed_dim=128,
            depth=1,
            num_heads=2,
            cond_token_dim=16,
            global_cond_dim=16,
            global_cond_type="adaLN",
            local_add_cond_dim=3,
            num_memory_tokens=2,
            zero_init_branch_outputs=False,
        )
        backbone = ConditionedDiffusionModelWrapper(
            dit,
            None,
            io_channels=2,
            sample_rate=8,
            min_input_length=1,
            diffusion_objective="rectified_flow",
            cross_attn_cond_ids=["prompt", "seconds_total"],
            global_cond_ids=["seconds_total"],
            local_add_cond_ids=["inpaint_mask", "inpaint_masked_input"],
        )
        backbone.eval().requires_grad_(False)
        module = PromptDecomposer(adapter_config())
        with (
            patch.object(transformer, "flash_attn_func", None),
            patch.object(transformer, "flash_attn_varlen_func", None),
        ):
            pred, _ = PrismPredictor(backbone, module, 2, True)(
                torch.randn(2, 2, 7),
                torch.tensor([0.1, 0.9]),
                conditioning(),
                torch.ones(2, 7, dtype=torch.bool),
                torch.linspace(0, 1, 7),
                torch.ones(2),
            )
            pred.square().mean().backward()
        self.assertTrue(torch.isfinite(pred).all())
        self.assertGreater(float(module.queries.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))

    def test_cfg_zero_pair_uses_same_seed_and_no_annotation(self):
        backbone, module = MockBackbone(), PromptDecomposer(adapter_config())
        config = TrainConfig(
            adapter=adapter_config(),
            sample_rate=8,
            downsampling_ratio=2,
            latent_alignment=2,
            headroom_seconds=1,
            precision="fp32",
        )
        seen = []

        def encode(_, records, device, native=False):
            seen.extend(records)
            return {
                "prompt": [torch.ones(1, 5, 16), torch.ones(1, 5)],
                "seconds_total": [torch.ones(1, 1, 16), torch.ones(1, 1)],
            }

        with patch("stable_audio_3.prism.system.encode_conditioning", encode):
            a, _ = generate(
                backbone, module, "A then B", 1, config, steps=2, cfg_scale=0, seed=7
            )
            b, _ = generate(
                backbone,
                module,
                "A then B",
                1,
                config,
                steps=2,
                cfg_scale=0,
                seed=7,
                baseline=True,
            )
        torch.testing.assert_close(a, b)
        self.assertEqual(a.shape, (1, 2, 8))
        self.assertTrue(all(not r["events"] for r in seen))


class TemporalAndDataTests(unittest.TestCase):
    def test_scalar_annotation_is_not_interval(self):
        annotation = {"event": {"siren": [1.2, 0.7]}}
        for task in ("duration", "frequency"):
            self.assertIsNone(parse_events(annotation, task, 10)[0]["intervals"])
        with self.assertRaises(ValueError):
            parse_events(annotation, "timestamp", 10)

    def test_invalid_interval_fails(self):
        for interval in ([3, 2], [0, 11], [-1, 2]):
            with self.assertRaises(ValueError):
                parse_events({"event": {"A": [interval]}}, "ordering", 10)

    def test_latent_masks_short_clip_and_headroom(self):
        d, signal, attention, positions, _ = latent_geometry(
            [{"duration": 1.5}], 8, 8, 2, 0.5, "cpu"
        )
        self.assertEqual(int(signal.sum()), 6)
        self.assertEqual(int(attention.sum()), 8)
        self.assertEqual(float(d[0]), 1.5)
        self.assertGreater(float(positions.max()), 1)

    def test_pit_invariant_to_event_order_and_no_false_interval_supervision(self):
        logits = torch.tensor(
            [[[5.0, 5, -5, -5], [-5.0, -5, 5, 5], [-5.0, -5, -5, -5]]],
            requires_grad=True,
        )
        components = torch.randn(1, 3, 2, 16, requires_grad=True)
        sample = {
            "uid": "x",
            "has_intervals": True,
            "latent_times": torch.arange(4),
            "events": [{"intervals": [[0, 2]]}, {"intervals": [[2, 4]]}],
        }
        a, _ = temporal_matching_loss(components, logits, [sample], torch.ones(1, 4))
        b, _ = temporal_matching_loss(
            components,
            logits,
            [{**sample, "events": sample["events"][::-1]}],
            torch.ones(1, 4),
        )
        torch.testing.assert_close(a, b)
        self.assertLess(float(a), 0.01)
        a.backward()
        self.assertIsNotNone(logits.grad)
        none, _ = temporal_matching_loss(
            components, logits, [{**sample, "has_intervals": False}], torch.ones(1, 4)
        )
        self.assertEqual(float(none), 0)

    def test_resample_stereo_whole_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            audio = torch.sin(torch.arange(1600) * 0.1).numpy()
            sf.write(path / "x.wav", audio, 16000)
            record = {
                "uid": "x",
                "duration": 0.1,
                "sample_rate": 16000,
                "events": [],
                "audio_path": str(path / "x.wav"),
            }
            (path / "train.jsonl").write_text(json.dumps(record) + "\n")
            config = TrainConfig(clip_seconds=0.1, headroom_seconds=0)
            waveform = AudioTimeDataset(path / "train.jsonl", config)[0]["audio"]
            self.assertEqual(waveform.shape, (2, 8192))
            torch.testing.assert_close(waveform[0], waveform[1])
            self.assertTrue(waveform[:, :4410].abs().sum() > 0)
            self.assertTrue((waveform[:, 4410:] == 0).all())
            with self.assertRaises(ValueError):
                AudioTimeDataset(
                    path / "train.jsonl", replace(config, clip_seconds=0.01)
                )

    def test_validation_subsample_covers_tasks(self):
        records = [
            {"task": task, "uid": f"{task}:{i}"}
            for task in ("duration", "frequency", "ordering", "timestamp")
            for i in range(10)
        ]
        chosen = validation_indices(records, 4, 1)
        self.assertEqual(len({records[i]["task"] for i in chosen}), 4)


class ResumeAndMetricsTests(unittest.TestCase):
    def test_checkpoint_resume_optimizer_and_rng(self):
        torch.manual_seed(4)
        random.seed(4)
        config = TrainConfig(adapter=adapter_config())
        module = PromptDecomposer(config.adapter)
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)

        def step(m, opt):
            opt.zero_grad()
            z = m(torch.randn(1, 4, 16), torch.ones(1, 4), torch.rand(1))["components"]
            (z.square().mean() + orthogonality_loss(z)).backward()
            opt.step()

        step(module, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            position, fingerprints = (
                {"epoch": 0, "next_batch": 8, "step": 1},
                {"data": "abc", "backbone": "def"},
            )
            save_checkpoint(path, module, optimizer, config, position, fingerprints)
            expected_random = random.random()
            step(module, optimizer)
            expected = copy.deepcopy(module.state_dict())
            restored = PromptDecomposer(config.adapter)
            opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            actual_position = load_checkpoint(path, restored, opt, config, fingerprints)
            self.assertEqual(actual_position, position)
            self.assertEqual(random.random(), expected_random)
            step(restored, opt)
            for key in expected:
                torch.testing.assert_close(
                    expected[key], restored.state_dict()[key], atol=0, rtol=0
                )
            with self.assertRaises(ValueError):
                load_checkpoint(
                    path, restored, opt, config, {"data": "wrong", "backbone": "def"}
                )

    def test_validation_restores_rng(self):
        torch.manual_seed(9)
        state = torch.get_rng_state()
        with validation_rng(10, "cpu"):
            torch.rand(13)
        torch.testing.assert_close(torch.get_rng_state(), state)

    def test_missing_events_and_reversed_order_are_penalized(self):
        row = {
            "uid": "x",
            "task": "ordering",
            "duration": 10,
            "events": [
                {"name": "A", "intervals": [[1, 2]]},
                {"name": "B", "intervals": [[5, 6]]},
            ],
        }
        missing = score_detections(
            [row], [{"uid": "x", "events": [{"name": "A", "start": 1, "end": 2}]}]
        )
        self.assertEqual(missing["event_type_recall"], 0.5)
        self.assertEqual(missing["ordering_pair_error_rate"], 1)
        reverse = score_detections(
            [row],
            [
                {
                    "uid": "x",
                    "events": [
                        {"name": "A", "start": 7, "end": 8},
                        {"name": "B", "start": 2, "end": 3},
                    ],
                }
            ],
        )
        self.assertEqual(reverse["all_events_present_rate"], 1)
        self.assertEqual(reverse["ordering_pair_error_rate"], 1)

    def test_segment_matching_and_detection_completeness(self):
        self.assertEqual(matched_intervals([[0, 1]], [[0, 1], [0, 1]]), 1)
        row = {
            "uid": "x",
            "task": "timestamp",
            "duration": 10,
            "events": [{"name": "A", "intervals": [[1, 2]]}],
        }
        score = score_detections(
            [row], [{"uid": "x", "events": [{"name": "A", "start": 1, "end": 2}]}]
        )
        self.assertEqual(score["timestamp_segment_iou_f1"], 1)
        with self.assertRaises(ValueError):
            score_detections([row], [])

    def test_duration_frequency_and_tied_order(self):
        duration = {
            "uid": "d",
            "task": "duration",
            "duration": 10,
            "events": [{"name": "A", "values": [2.0]}],
        }
        frequency = {
            "uid": "f",
            "task": "frequency",
            "duration": 10,
            "events": [{"name": "A", "values": [0.3, 0.4]}],
        }
        score = score_detections(
            [duration, frequency],
            [
                {"uid": "d", "events": [{"name": "A", "start": 1, "end": 2}]},
                {"uid": "f", "events": [{"name": "A", "start": 1, "end": 2}]},
            ],
        )
        self.assertEqual(score["duration_mae_seconds"], 1)
        self.assertEqual(score["frequency_mae"], 1)
        row = {
            "uid": "o",
            "task": "ordering",
            "duration": 10,
            "events": [
                {"name": "A", "intervals": [[5, 6]]},
                {"name": "B", "intervals": [[1, 2]]},
            ],
        }
        tied = score_detections(
            [row],
            [
                {
                    "uid": "o",
                    "events": [
                        {"name": "A", "start": 2, "end": 3},
                        {"name": "B", "start": 2, "end": 3},
                    ],
                }
            ],
        )
        self.assertEqual(tied["ordering_pair_error_rate"], 1)

    def test_semantic_matching_backpropagates_without_fake_times(self):
        components = torch.randn(1, 3, 2, 16, requires_grad=True)
        logits = torch.zeros(1, 3, 5, requires_grad=True)
        sample = {
            "uid": "frequency",
            "has_intervals": False,
            "events": [{"name": "A", "intervals": None}],
        }
        temporal, semantic = temporal_matching_loss(
            components,
            logits,
            [sample],
            torch.ones(1, 5),
            [torch.randn(1, 16)],
            0.1,
            0.1,
        )
        self.assertEqual(float(temporal), 0)
        semantic.backward()
        self.assertGreater(float(components.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
