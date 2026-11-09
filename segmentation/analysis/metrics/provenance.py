"""Collect source, input, and dataset provenance for metric measurements."""

import hashlib
import importlib.metadata
import os
from pathlib import Path
import re
import subprocess

import yaml

from segmentation.analysis.metrics.catalog import sha256_file


_PACKAGES = (
    "torch", "torchvision", "numpy", "pandas", "Pillow", "opencv-python",
    "scikit-image", "scipy", "tqdm",
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b[\w.-]*(?:password|passwd|secret|token|api[_-]?key|credential|private[_-]?key)\b\s*[:=]"
)
_SENSITIVE_VALUE = re.compile(
    r"(?i)(\b[\w.-]*(?:password|passwd|secret|token|api[_-]?key|credential|private[_-]?key)\b\s*[:=]\s*)"
    r'(?:"[^"]*"|\'[^\']*\'|[^\s,;]+)'
)


def sanitize_error(error):
    message = f"{type(error).__name__}: {error}"
    message = _SENSITIVE_VALUE.sub(r"\1[redacted]", message)
    return "".join(character for character in message if character.isprintable())[:500]


def _run_git(checkout, *arguments):
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_provenance(checkout):
    checkout = Path(checkout)
    if _run_git(checkout, "rev-parse", "--show-toplevel") is None:
        return {
            "commit": None,
            "tracked_diff": None,
            "relevant_untracked_source": [],
            "submodules": [],
            "missing": ["git source identity"],
        }
    diff = _run_git(checkout, "diff", "--binary", "HEAD")
    untracked = _run_git(checkout, "ls-files", "--others", "--exclude-standard") or ""
    source_files = []
    omitted_source = []
    for relative in untracked.splitlines():
        path = Path(relative)
        if path.suffix not in (".py", ".sh") or path.parts[0] not in ("segmentation", "scripts"):
            continue
        source = checkout / path
        if source.is_symlink():
            omitted_source.append(str(path))
            continue
        try:
            source.resolve().relative_to(checkout.resolve())
            contents = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            omitted_source.append(str(path))
            continue
        if _SENSITIVE_ASSIGNMENT.search(contents):
            omitted_source.append(str(path))
            continue
        source_files.append({
            "path": str(path),
            "sha256": hashlib.sha256(contents.encode()).hexdigest(),
            "content": contents,
        })
    submodules = _run_git(checkout, "submodule", "status") or ""
    tracked_diff = None
    if diff:
        tracked_diff = {
            "sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "content": None if _SENSITIVE_ASSIGNMENT.search(diff) else diff,
        }
        if tracked_diff["content"] is None:
            tracked_diff["content_omitted_for_sensitive_content"] = True
    return {
        "commit": _run_git(checkout, "rev-parse", "HEAD"),
        "tracked_diff": tracked_diff,
        "relevant_untracked_source": source_files,
        "sensitive_looking_untracked_source_omitted": omitted_source,
        "submodules": [line.strip() for line in submodules.splitlines() if line.strip()],
        "missing": [],
    }


def package_versions():
    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def file_identity(path, cache):
    absolute = str(Path(path).expanduser().resolve())
    if absolute not in cache:
        source = Path(absolute)
        if source.is_file():
            try:
                digest, size = sha256_file(source)
                cache[absolute] = {
                    "path": absolute,
                    "bytes": size,
                    "sha256": digest,
                    "status": "available",
                }
            except OSError as error:
                cache[absolute] = {
                    "path": absolute,
                    "bytes": None,
                    "sha256": None,
                    "status": "unreadable",
                    "error": str(error)[:200],
                }
        else:
            cache[absolute] = {
                "path": absolute,
                "bytes": None,
                "sha256": None,
                "status": "missing",
                "reason": "Input file does not exist at measurement initialization.",
            }
    return dict(cache[absolute])


def part_mapping_metadata(dataset_root, data_type, checkout, cache):
    annotation_provenance = "unknown"
    checkout = Path(checkout)
    if data_type == "cityscapes":
        mapping_file = checkout / "segmentation" / "data" / "cityscapes_parts.py"
        human_parts = {"1": "torso", "2": "head", "3": "arm", "4": "leg"}
        vehicle_parts = {
            "1": "window",
            "2": "wheel",
            "3": "light",
            "4": "license plate",
            "5": "chassis",
        }
        definitions = {
            "person": {"semantic_ids": [24], "part_id_to_name": human_parts},
            "rider": {"semantic_ids": [25], "part_id_to_name": human_parts},
            "car": {"semantic_ids": [26], "part_id_to_name": vehicle_parts},
            "truck": {"semantic_ids": [27], "part_id_to_name": vehicle_parts},
            "bus": {"semantic_ids": [28], "part_id_to_name": vehicle_parts},
        }
        mapping_identity = file_identity(mapping_file, cache)
        mapping_identity["identity_kind"] = "official Cityscapes filter implementation"
        return {
            "mapping_identity": mapping_identity,
            "class_dependent_part_definitions": definitions,
            "annotation_provenance": annotation_provenance,
        }

    candidates = [Path(dataset_root) / "parts.yaml", Path(dataset_root).parent / "parts.yaml"]
    if data_type == "pascal":
        configured_source = os.environ.get("SOURCE_DATA_PATH_PASCAL")
        if configured_source:
            candidates.append(Path(configured_source).expanduser() / "parts.yaml")
    mapping_file = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if mapping_file is None:
        return {
            "mapping_identity": {
                "status": "unknown",
                "reason": "No configured parts.yaml mapping specification was available.",
            },
            "class_dependent_part_definitions": None,
            "annotation_provenance": annotation_provenance,
        }
    try:
        mapping = yaml.safe_load(mapping_file.read_text(encoding="utf-8"))
        class_parts = mapping["scene_class2part_classes"]
        countable_groups = mapping.get("countable_pids_groupings", {})
        if not isinstance(class_parts, dict) or not isinstance(countable_groups, dict):
            raise ValueError("unexpected Pascal part mapping schema")
        definitions = {
            str(class_name): {
                "semantic_class_id": class_index + 1,
                "part_ids": {str(part_name): part_index + 1 for part_index, part_name in enumerate(parts)},
                "countable_pid_groups": countable_groups.get(class_name, {}),
            }
            for class_index, (class_name, parts) in enumerate(class_parts.items())
        }
    except (OSError, UnicodeDecodeError, yaml.YAMLError, KeyError, TypeError, ValueError) as error:
        return {
            "mapping_identity": {
                "status": "invalid",
                "path": str(mapping_file),
                "reason": str(error)[:200],
            },
            "class_dependent_part_definitions": None,
            "annotation_provenance": annotation_provenance,
        }
    return {
        "mapping_identity": file_identity(mapping_file, cache),
        "class_dependent_part_definitions": definitions,
        "annotation_provenance": annotation_provenance,
    }


def dataset_provenance(dataset, data_type, checkout, cache):
    sample_ids = list(dataset.img_ids)
    annotations = Path(dataset.annotations_dir).expanduser().resolve()
    dataset_root = annotations.parent.parent
    part_dir = dataset_root / "annotations_PIDS" / annotations.name
    part_available = [
        sample_id for sample_id in sample_ids if (part_dir / f"{sample_id}.npy").is_file()
    ]
    dataset_fields = {
        "split": "val",
        "processed_root": str(dataset_root),
        "annotations_directory": str(annotations),
        "part_annotations_directory": str(part_dir),
        "semantic_id_annotations_directory": str(dataset_root / "annotations_SIDS" / annotations.name),
        "candidate_count": len(sample_ids),
        "part_annotation_count": len(part_available),
        "part_annotation_coverage": len(part_available) / len(sample_ids) if sample_ids else None,
        "semantic_id_annotation_count": sum(
            (dataset_root / "annotations_SIDS" / annotations.name / f"{sample_id}.npy").is_file()
            for sample_id in sample_ids
        ),
        "identity": "partial",
    }
    warnings = [
        "Dataset identity is partial; image paths are not content fingerprints.",
        f"{len(sample_ids) - len(part_available)} candidates lack part annotations."
        if len(sample_ids) != len(part_available) else "Every candidate has a part annotation.",
    ]
    parts_fields = part_mapping_metadata(dataset_root, data_type, checkout, cache)
    warnings.append(
        "The preprocessing provenance linking part annotations to the mapping is unknown."
    )
    return sample_ids, dataset_fields, parts_fields, warnings, len(part_available) != len(sample_ids)
