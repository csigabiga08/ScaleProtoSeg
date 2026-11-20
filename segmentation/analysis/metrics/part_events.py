"""Within-class profile similarity and positive-event overlap from binary hits."""

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median


@dataclass
class EventProfile:
    class_id: int
    prototype_id: int
    observed: dict
    positive: dict

    @property
    def consistent(self):
        return any(len(self.positive[part]) / len(images) > 0.8 for part, images in self.observed.items() if images)


def event_profiles(record, rows):
    """Keep availability separate from hits; count each image at most once."""
    columns = [
        column
        for column in record["protocol"]["observation_columns"]
        if column.startswith("part_") and column[5:].isdigit() and int(column[5:]) > 0
    ]
    valid = set(record["protocol"]["parts"]["valid_part_ids"])
    columns = [column for column in columns if int(column[5:]) in valid]
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Expected unique positive part columns")
    profiles, seen, classes = {}, set(), {}
    for row in rows:
        cid, pid = int(row["class_id"]), int(row["prototype_id"])
        image = row["sample_id"]
        if (image, cid, pid) in seen:
            raise ValueError("Repeated image/class/prototype observation")
        seen.add((image, cid, pid))
        if pid in classes and classes[pid] != cid:
            raise ValueError("A prototype must belong to exactly one class")
        classes[pid] = cid
        if (cid, pid) not in profiles:
            profiles[cid, pid] = EventProfile(cid, pid, defaultdict(set), defaultdict(set))
        profile = profiles[cid, pid]
        for column in columns:
            value = row[column]
            if value is None or value == "" or str(value).lower() == "nan":
                continue
            number = float(value)
            if number not in (0, 1):
                raise ValueError("Expected a binary part hit or missing value")
            part = int(column[5:])
            profile.observed[part].add(image)
            if number:
                profile.positive[part].add(image)
    return profiles


def pearson(left, right):
    if len(left) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    x = [value - left_mean for value in left]
    y = [value - right_mean for value in right]
    xx, yy = math.fsum(value * value for value in x), math.fsum(value * value for value in y)
    if not xx or not yy:
        return None
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(x, y)) / math.sqrt(xx * yy)))


def compare_events(left, right, *, minimum_support=1, minimum_parts=2):
    """Compare mean profiles on shared parts and events on shared image/part cells.

    Support filtering uses joint image availability. Native means retain each
    prototype's own observed images; matched means use the Jaccard event mask.
    """
    if left.class_id != right.class_id:
        raise ValueError("Part IDs are class-local; cross-class comparisons are undefined")
    if minimum_support < 1 or minimum_parts < 2:
        raise ValueError("Expected positive support and at least two parts")
    shared = sorted(left.observed.keys() & right.observed.keys())
    supports = {part: left.observed[part] & right.observed[part] for part in shared}
    included = [part for part in shared if len(supports[part]) >= minimum_support]
    native_left, native_right, matched_left, matched_right, counts = [], [], [], [], []
    intersection = union = left_hits = right_hits = 0
    for part in included:
        images = supports[part]
        a, b = left.positive[part] & images, right.positive[part] & images
        counts.append(len(images))
        left_hits += len(a)
        right_hits += len(b)
        intersection += len(a & b)
        union += len(a | b)
        native_left.append(len(left.positive[part]) / len(left.observed[part]))
        native_right.append(len(right.positive[part]) / len(right.observed[part]))
        matched_left.append(len(a) / len(images))
        matched_right.append(len(b) / len(images))
    own_cells = sum(len(left.observed[part]) + len(right.observed[part]) for part in included)
    joint_cells = sum(counts)
    return {
        "shared_parts": len(shared),
        "included_parts": len(included),
        "eligible": len(included) >= minimum_parts,
        "minimum_joint_support": min(counts) if counts else None,
        "median_joint_support": median(counts) if counts else None,
        "maximum_joint_support": max(counts) if counts else None,
        "joint_event_cells": joint_cells,
        "availability_dice": 2 * joint_cells / own_cells if own_cells else None,
        "rho": pearson(native_left, native_right),
        "matched_rho": pearson(matched_left, matched_right),
        "profile_mad": mean(abs(a - b) for a, b in zip(native_left, native_right)) if included else None,
        "left_positive_events": left_hits,
        "right_positive_events": right_hits,
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else None,
    }
