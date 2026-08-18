"""Generate a deterministic CycloneDX SBOM from project metadata and a lock.

The SBOM must describe the committed dependency graph, not the interpreter
that happens to run this script. In particular, editable installs can leak
absolute ``file://`` URLs into environment-generated SBOMs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
import uuid
from pathlib import Path
from typing import Any

_PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)==(?P<version>[^ ;\\]+)"
    r"(?:\s*;\s*(?P<marker>[^\\]+?))?\s*\\?\s*$"
)
_HASH_RE = re.compile(r"^\s+--hash=sha256:(?P<digest>[0-9a-f]+)(?:\s+\\)?\s*$")
_FORBIDDEN_PATHS = (
    "file://",
    "/Users/",
    "/home/runner",
    "/workspace/",
    "/workspaces/",
)


def _pep503(name: str) -> str:
    """Return the normalized package name used in a PyPI PURL."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_lock(lock_text: str) -> list[dict[str, Any]]:
    """Parse exact pins, markers, and hashes from a uv requirements export."""
    components: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in lock_text.splitlines():
        match = _PIN_RE.match(line)
        if match:
            if current is not None:
                components.append(current)
            current = {
                "name": match.group("name"),
                "version": match.group("version"),
                "marker": (match.group("marker") or "").strip(),
                "hashes": [],
            }
            continue

        if current is not None:
            hash_match = _HASH_RE.match(line)
            if hash_match:
                current["hashes"].append(hash_match.group("digest"))

    if current is not None:
        components.append(current)
    if not components:
        raise ValueError("No exact package pins found in requirements lock")
    return components


def _component(entry: dict[str, Any]) -> dict[str, Any]:
    purl = f"pkg:pypi/{_pep503(entry['name'])}@{entry['version']}"
    component: dict[str, Any] = {
        "bom-ref": purl,
        "name": entry["name"],
        "purl": purl,
        "type": "library",
        "version": entry["version"],
    }
    properties: list[dict[str, str]] = []
    if entry["marker"]:
        properties.append({"name": "uv:marker", "value": entry["marker"]})
    properties.extend({"name": "uv:sha256", "value": digest} for digest in entry["hashes"])
    if properties:
        component["properties"] = properties
    return component


def validate_sbom(sbom: dict[str, Any]) -> None:
    """Reject host-specific paths before an SBOM can be written."""
    serialized = json.dumps(sbom, ensure_ascii=True, sort_keys=True)
    for forbidden in _FORBIDDEN_PATHS:
        if forbidden in serialized:
            raise ValueError(f"SBOM contains forbidden host path marker: {forbidden!r}")


def build_sbom(project_path: Path, lock_path: Path) -> dict[str, Any]:
    """Build a CycloneDX document from exactly two committed input files."""
    project_bytes = project_path.read_bytes()
    lock_bytes = lock_path.read_bytes()
    project = tomllib.loads(project_bytes.decode("utf-8"))
    project_metadata = project["project"]
    project_name = str(project_metadata["name"])
    project_version = str(project_metadata["version"])

    entries = _parse_lock(lock_bytes.decode("utf-8"))
    components = sorted(
        (_component(entry) for entry in entries),
        key=lambda component: (
            component["name"].lower(),
            component["version"],
            component["bom-ref"],
        ),
    )
    refs_by_name: dict[str, list[str]] = {}
    for component in components:
        refs_by_name.setdefault(_pep503(component["name"]), []).append(component["bom-ref"])

    direct_names = {
        _pep503(str(requirement).split("[", 1)[0].split(">", 1)[0].split("=", 1)[0])
        for requirement in project_metadata.get("dependencies", [])
    }
    app_ref = f"pkg:pypi/{_pep503(project_name)}@{project_version}"
    source_digest = hashlib.sha256(project_bytes + b"\0" + lock_bytes).hexdigest()
    serial = f"urn:uuid:{uuid.UUID(bytes=bytes.fromhex(source_digest[:32]), version=5)}"
    sbom: dict[str, Any] = {
        "$schema": "http://cyclonedx.org/schema/bom-1.6.schema.json",
        "bomFormat": "CycloneDX",
        "components": components,
        "dependencies": [
            {
                "dependsOn": sorted(
                    ref for name in direct_names for ref in refs_by_name.get(name, [])
                ),
                "ref": app_ref,
            }
        ],
        "metadata": {
            "component": {
                "bom-ref": app_ref,
                "name": project_name,
                "type": "application",
                "version": project_version,
            },
            "properties": [
                {
                    "name": "atlas:pyproject-sha256",
                    "value": hashlib.sha256(project_bytes).hexdigest(),
                },
                {
                    "name": "atlas:requirements-lock-sha256",
                    "value": hashlib.sha256(lock_bytes).hexdigest(),
                },
            ],
        },
        "serialNumber": serial,
        "specVersion": "1.6",
        "version": 1,
    }
    validate_sbom(sbom)
    return sbom


def write_sbom(project_path: Path, lock_path: Path, output_path: Path) -> None:
    """Write a canonical, newline-terminated SBOM."""
    sbom = build_sbom(project_path, lock_path)
    output_path.write_text(json.dumps(sbom, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--lock", type=Path, default=Path("requirements.lock"))
    parser.add_argument("--output", type=Path, default=Path("sbom.json"))
    args = parser.parse_args()
    write_sbom(args.project, args.lock, args.output)


if __name__ == "__main__":
    main()
