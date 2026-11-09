"""Population counts and ordered sample ID digests shared by writers and validators."""

import hashlib
import json


def population_counts(samples, observation_count):
    outcomes = [sample["outcome"] for sample in samples]
    return {
        "candidates": len(outcomes),
        "attempted": sum(
            value in ("processed_no_observations", "contributed", "failed")
            for value in outcomes
        ),
        "processed": sum(
            value in ("processed_no_observations", "contributed")
            for value in outcomes
        ),
        "contributing": outcomes.count("contributed"),
        "failed": outcomes.count("failed"),
        "excluded": outcomes.count("excluded"),
        "not_attempted": outcomes.count("not_attempted"),
        "processed_no_observations": outcomes.count("processed_no_observations"),
        "observations": observation_count,
    }


def ordered_ids_sha256(sample_ids):
    serialized = json.dumps(
        list(sample_ids), separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(serialized).hexdigest()
