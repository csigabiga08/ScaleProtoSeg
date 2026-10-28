"""Repair the legacy Python version check in the active environment."""

from importlib.metadata import distribution
from pathlib import Path
import sys


def main() -> None:
    package = distribution("panoptic-parts")
    source_path = Path(package.locate_file("panoptic_parts/specs/dataset_spec.py"))
    source = source_path.read_bytes()
    newline = b"\r\n" if b"\r\n" in source else b"\n"
    old_check = newline.join([
        b"import platform",
        b"version = platform.python_version()",
        b"if float(version[:3]) <= 3.6:",
    ])
    new_check = newline.join([
        b"import sys",
        b"if sys.version_info < (3, 7):",
    ])

    if new_check in source:
        print(f"Python version check already repaired: {source_path}")
        return
    if source.count(old_check) != 1:
        raise SystemExit(
            "The expected legacy version check was not found; no files changed. "
            "Check whether this panoptic-parts release still needs the repair."
        )

    backup_dir = Path(sys.prefix) / "compat-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / (
        f"panoptic-parts-{package.version}-dataset_spec.py.before-version-fix"
    )
    if not backup_path.exists():
        backup_path.write_bytes(source)
    source_path.write_bytes(source.replace(old_check, new_check, 1))
    print(f"Repaired Python version check: {source_path}")
    print(f"Original file backed up to: {backup_path}")


if __name__ == "__main__":
    main()
