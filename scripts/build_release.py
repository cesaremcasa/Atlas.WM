"""Build and validate the v4.0.1 release staging artifacts.

The build uses the committed lock as a hash-verified constraint for PEP 517
build dependencies, fixes ``SOURCE_DATE_EPOCH``, canonicalizes archive
metadata, emits the lock-derived SBOM, and writes deterministic SHA256 sums.
It never publishes or tags a release.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

from generate_sbom import write_sbom

VERSION = "4.0.1"
WHEEL_NAME = f"atlas_wm-{VERSION}-py3-none-any.whl"
SDIST_NAME = f"atlas_wm-{VERSION}.tar.gz"
FORBIDDEN_TEXT = ("file://", "/Users/", "/home/runner", "/workspace", "/workspaces")
FORBIDDEN_MEMBERS = (
    "AI Search/",
    "archive/",
    "artifacts/",
    "checkpoints/",
    "paper/",
)
FORBIDDEN_SUFFIXES = (".npy", ".npz", ".pdf", ".safetensors", ".pt", ".pth")


def _run_build(repo_root: Path, output_dir: Path) -> None:
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = "0"
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--sdist",
            "--clear",
            "--no-create-gitignore",
            "--out-dir",
            str(output_dir),
            "--build-constraints",
            str(repo_root / "requirements.lock"),
            "--require-hashes",
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )


def _canonicalize_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as source:
        members = [
            (info.filename, source.read(info.filename), info.external_attr)
            for info in source.infolist()
        ]
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data, external_attr in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = external_attr
            archive.writestr(info, data)
    path.write_bytes(output.getvalue())


def _canonicalize_sdist(path: Path) -> None:
    with gzip.open(path, "rb") as compressed:
        source_bytes = compressed.read()
    output_tar = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(source_bytes), mode="r:") as source:
        members = sorted(source.getmembers(), key=lambda member: member.name)
        with tarfile.open(fileobj=output_tar, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for original in members:
                member = copy.copy(original)
                member.mtime = 0
                member.uid = 0
                member.gid = 0
                member.uname = ""
                member.gname = ""
                member.pax_headers = {}
                if member.isfile():
                    data = source.extractfile(original)
                    if data is None:
                        raise ValueError(f"unable to read sdist member: {original.name}")
                    payload = data.read()
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
                else:
                    archive.addfile(member)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(output_tar.getvalue())
    path.write_bytes(output.getvalue())


def _members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with (
        gzip.open(path, "rb") as compressed,
        tarfile.open(fileobj=compressed, mode="r:") as archive,
    ):
        return [member.name for member in archive.getmembers()]


def _validate_archive(path: Path) -> None:
    member_names = _members(path)
    for name in member_names:
        if name.startswith(FORBIDDEN_MEMBERS) or name.endswith(FORBIDDEN_SUFFIXES):
            raise ValueError(f"forbidden release member {name!r} in {path.name}")
    raw = path.read_bytes()
    for marker in FORBIDDEN_TEXT:
        if marker.encode() in raw:
            raise ValueError(f"host-specific path marker {marker!r} in {path.name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(paths: Iterable[Path], output: Path) -> None:
    lines = [f"{_sha256(path)}  {path.name}" for path in sorted(paths, key=lambda path: path.name)]
    output.write_text("\n".join(lines) + "\n")


def validate_staging(output_dir: Path) -> None:
    expected = {WHEEL_NAME, SDIST_NAME, "sbom.json", "SHA256SUMS"}
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual != expected:
        raise ValueError(f"unexpected release staging contents: {sorted(actual)}")
    _validate_archive(output_dir / WHEEL_NAME)
    _validate_archive(output_dir / SDIST_NAME)
    if any(marker.encode() in (output_dir / "sbom.json").read_bytes() for marker in FORBIDDEN_TEXT):
        raise ValueError("host-specific path marker found in staged SBOM")
    checksum_lines = (output_dir / "SHA256SUMS").read_text().splitlines()
    expected_lines = [
        f"{_sha256(output_dir / name)}  {name}"
        for name in sorted((WHEEL_NAME, SDIST_NAME, "sbom.json"))
    ]
    if checksum_lines != expected_lines:
        raise ValueError("SHA256SUMS does not match staged artifacts")


def build_artifacts(repo_root: Path, output_dir: Path) -> dict[str, str]:
    """Build one clean staging directory and return artifact hashes."""
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir == repo_root or repo_root in output_dir.parents:
        raise ValueError("release staging must be outside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_build(repo_root, output_dir)
    _canonicalize_wheel(output_dir / WHEEL_NAME)
    _canonicalize_sdist(output_dir / SDIST_NAME)
    write_sbom(
        repo_root / "pyproject.toml", repo_root / "requirements.lock", output_dir / "sbom.json"
    )
    _write_checksums(
        [output_dir / WHEEL_NAME, output_dir / SDIST_NAME, output_dir / "sbom.json"],
        output_dir / "SHA256SUMS",
    )
    validate_staging(output_dir)
    return {path.name: _sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    hashes = build_artifacts(args.repo_root, args.output_dir)
    for name, digest in hashes.items():
        print(f"{digest}  {name}")


if __name__ == "__main__":
    main()
