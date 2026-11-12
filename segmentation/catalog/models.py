"""Checkpoint registry with transactional UUID, name and content uniqueness."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from segmentation.catalog import CatalogError, resolve_catalog_root, sha256_file
from segmentation.catalog.architectures import validate_config
from segmentation.catalog.schema import Artifact, Text


class ModelRegistration(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    model_id: Text
    name: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
    architecture: Text
    schema_version: int = 1
    config: dict
    dataset: Text
    seed: Optional[int] = None
    tags: dict[str, str] = Field(default_factory=dict)
    checkpoint: Artifact
    created_at_utc: Text

    @model_validator(mode="after")
    def architecture_config(self):
        if str(uuid.UUID(self.model_id)) != self.model_id:
            raise ValueError("model_id must be a canonical UUID")
        try:
            uuid.UUID(self.name)
        except ValueError:
            pass
        else:
            raise ValueError("A model name must not itself be a UUID")
        self.config = validate_config(self.architecture, self.schema_version, self.config)
        return self


def _root(root):
    return Path(root) if root is not None else resolve_catalog_root()


@contextmanager
def _connect(root, *, write=False):
    database = _root(root) / "models.sqlite3"
    if not write and not database.is_file():
        raise CatalogError("No models registered; register a checkpoint before measuring it")
    try:
        connection = sqlite3.connect(
            str(database) if write else database.as_uri() + "?mode=ro",
            uri=not write,
            timeout=30,
        )
    except sqlite3.Error as error:
        raise CatalogError(f"Cannot access model registry: {error}") from error
    try:
        with connection:
            if write:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS models "
                    "(model_id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL UNIQUE, record TEXT NOT NULL)"
                )
            yield connection
    except sqlite3.IntegrityError:
        raise
    except sqlite3.Error as error:
        raise CatalogError(f"Cannot access model registry: {error}") from error
    finally:
        connection.close()


def model_snapshot(registration):
    return {
        "registration_id": registration["model_id"],
        "name": registration["name"],
        "label": registration["name"],
        "architecture": registration["architecture"],
        "schema_version": registration["schema_version"],
        "config": registration["config"],
        "training_stage": registration["config"]["stage"],
        "dataset": registration["dataset"],
        "seed": registration["seed"],
        "tags": registration["tags"],
        "inputs": {"prototype_checkpoint": registration["checkpoint"]},
    }


def list_models(root=None):
    root = _root(root).resolve()
    if not (root / "models.sqlite3").is_file():
        return []
    with _connect(root) as connection:
        return [
            ModelRegistration.model_validate_json(row[0]).model_dump()
            for row in connection.execute("SELECT record FROM models ORDER BY name")
        ]


def find_model(reference, root=None, *, verify=False):
    with _connect(_root(root).resolve()) as connection:
        rows = connection.execute(
            "SELECT record FROM models WHERE model_id=? OR name=?",
            (str(reference), str(reference)),
        ).fetchall()
    if len(rows) != 1:
        raise CatalogError(f"Registered model not found or ambiguous: {reference}")
    record = ModelRegistration.model_validate_json(rows[0][0]).model_dump()
    if verify:
        checkpoint = record["checkpoint"]
        try:
            digest, size = sha256_file(checkpoint["path"])
        except OSError as error:
            raise CatalogError(f"Checkpoint is unavailable for {record['name']}: {error}") from error
        if (digest, size) != (checkpoint["sha256"], checkpoint["bytes"]):
            raise CatalogError(f"Registered checkpoint changed: {record['name']}")
    return record


def register_model(
    *, name, architecture, config, dataset, checkpoint, seed=None, tags=None, schema_version=1, root=None
):
    root = _root(root).resolve()
    path = Path(checkpoint).expanduser().resolve()
    digest, size = sha256_file(path)
    record = ModelRegistration(
        model_id=str(uuid.uuid4()),
        name=name,
        architecture=architecture,
        config=config,
        dataset=dataset,
        schema_version=schema_version,
        seed=seed,
        tags=tags or {},
        checkpoint=Artifact(path=str(path), sha256=digest, bytes=size),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    ).model_dump()
    try:
        with _connect(root, write=True) as connection:
            connection.execute(
                "INSERT INTO models VALUES (?,?,?,?)",
                (record["model_id"], name, digest, json.dumps(record, allow_nan=False)),
            )
    except sqlite3.IntegrityError as error:
        raise CatalogError(f"Model name or checkpoint hash is already registered: {name}") from error
    return record


def relocate_model(reference, checkpoint, root=None):
    root = _root(root).resolve()
    record = find_model(reference, root)
    path = Path(checkpoint).expanduser().resolve()
    digest, size = sha256_file(path)
    if (digest, size) != (record["checkpoint"]["sha256"], record["checkpoint"]["bytes"]):
        raise CatalogError("Relocation requires exactly the registered checkpoint contents")
    record["checkpoint"]["path"] = str(path)
    with _connect(root, write=True) as connection:
        connection.execute("UPDATE models SET record=? WHERE model_id=?", (json.dumps(record), record["model_id"]))
    return record
