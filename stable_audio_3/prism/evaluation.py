"""Metrics consuming INDEPENDENT audio event detections, never adapter gate logits.

These are transparent diagnostic metrics, not a reproduction of STEAM's detector.
Detector event names must use AudioTime's event vocabulary (or be mapped beforehand).
"""

from collections import defaultdict


def interval_iou(a, b):
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return intersection / max(1e-12, max(a[1], b[1]) - min(a[0], b[0]))


def matched_intervals(reference, predicted, threshold=0.5):
    edges = [
        [
            j
            for j, target in enumerate(reference)
            if interval_iou(interval, target) >= threshold
        ]
        for interval in predicted
    ]
    assigned = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in range(len(predicted)))


def union_duration(intervals):
    total, current = 0.0, None
    for start, end in sorted(intervals):
        if current is None:
            current = [start, end]
        elif start <= current[1]:
            current[1] = max(current[1], end)
        else:
            total += current[1] - current[0]
            current = [start, end]
    return total + (current[1] - current[0] if current else 0.0)


def score_detections(records, detections, threshold=0.5):
    by_uid = {}
    for detection in detections:
        if detection["uid"] in by_uid:
            raise ValueError(
                "Duplicate detection UID; score baseline and PRISM separately"
            )
        by_uid[detection["uid"]] = detection
    missing = {r["uid"] for r in records} - by_uid.keys()
    extra = by_uid.keys() - {r["uid"] for r in records}
    if missing or extra:
        raise ValueError(
            f"Detection/manifest mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    counts = defaultdict(float)
    for row in records:
        predicted = defaultdict(list)
        for event in by_uid[row["uid"]]["events"]:
            start, end = float(event["start"]), float(event["end"])
            if not (0 <= start < end <= row["duration"] + 0.02):
                raise ValueError(f"Invalid detection interval for {row['uid']}")
            predicted[event["name"]].append([start, end])
        required = {e["name"] for e in row["events"]}
        present = required & predicted.keys()
        counts["clips"] += 1
        counts["required_event_types"] += len(required)
        counts["present_event_types"] += len(present)
        counts["all_events_present_clips"] += len(present) == len(required)
        counts["unexpected_event_types"] += len(predicted.keys() - required)
        for event in row["events"]:
            intervals = predicted[event["name"]]
            if row["task"] == "duration":
                counts["duration_abs_error_seconds"] += abs(
                    union_duration(intervals) - sum(event["values"])
                )
                counts["duration_targets"] += 1
            if row["task"] == "frequency":
                counts["frequency_abs_error"] += abs(
                    len(intervals) - len(event["values"])
                )
                counts["frequency_targets"] += 1
        if row["task"] == "ordering":
            for i, first in enumerate(row["events"]):
                for second in row["events"][i + 1 :]:
                    a = min(x[0] for x in first["intervals"])
                    b = min(x[0] for x in second["intervals"])
                    if a == b:
                        continue
                    counts["ordering_pairs"] += 1
                    left, right = predicted[first["name"]], predicted[second["name"]]
                    correct = bool(left and right) and (
                        (min(x[0] for x in left) - min(x[0] for x in right)) * (a - b)
                        > 0
                    )
                    counts["ordering_pair_errors"] += not correct
        if row["task"] == "timestamp":
            for name in required | predicted.keys():
                reference = next(
                    (e["intervals"] for e in row["events"] if e["name"] == name), []
                )
                n = matched_intervals(reference, predicted[name], threshold)
                counts["timestamp_tp"] += n
                counts["timestamp_fp"] += len(predicted[name]) - n
                counts["timestamp_fn"] += len(reference) - n
    if not counts["clips"]:
        raise ValueError("No evaluation records")
    result = {
        "event_type_recall": counts["present_event_types"]
        / counts["required_event_types"],
        "all_events_present_rate": counts["all_events_present_clips"] / counts["clips"],
        "unexpected_event_types_per_clip": counts["unexpected_event_types"]
        / counts["clips"],
        "raw_counts": dict(counts),
        "timestamp_iou_threshold": threshold,
        "note": "Independent detector diagnostics; not official STEAM and not model-gate scores",
    }
    for name, numerator, denominator in (
        ("duration_mae_seconds", "duration_abs_error_seconds", "duration_targets"),
        ("frequency_mae", "frequency_abs_error", "frequency_targets"),
        ("ordering_pair_error_rate", "ordering_pair_errors", "ordering_pairs"),
    ):
        result[name] = (
            counts[numerator] / counts[denominator] if counts[denominator] else None
        )
    denom = 2 * counts["timestamp_tp"] + counts["timestamp_fp"] + counts["timestamp_fn"]
    result["timestamp_segment_iou_f1"] = (
        2 * counts["timestamp_tp"] / denom if denom else None
    )
    return result
