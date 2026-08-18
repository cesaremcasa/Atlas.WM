"""Build and validate v4.0.1 release staging artifacts without publishing.

Only the package source allowlist is copied into a fresh temporary build
context. Archives are checked before extraction, extracted into a private
temporary directory, scanned for host paths/secrets, and canonicalized so the
same inputs produce the same bytes across platforms.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable

from generate_sbom import write_sbom

VERSION = "4.0.1"
WHEEL_NAME = f"atlas_wm-{VERSION}-py3-none-any.whl"
SDIST_NAME = f"atlas_wm-{VERSION}.tar.gz"
SOURCE_ALLOWLIST = ("pyproject.toml", "README.md", "LICENSE", "CHANGELOG.md", "src/atlas_wm")
FORBIDDEN_TEXT = (
    "file://",
    "/Users/",
    "/home/runner",
    "/workspace",
    "/workspaces",
    "/private/tmp/",
    "/tmp/",
    "/var/folders/",
    "-----BEGIN ",
    "AKIA",
    "ghp_",
    "sk-",
)
FORBIDDEN_MEMBERS = (
    "AI Search/",
    "archive/",
    "artifacts/",
    "checkpoints/",
    "datasets/",
    "experiments/",
    "paper/",
    "papers/",
    "tests/",
)
FORBIDDEN_SUFFIXES = (".npy", ".npz", ".pdf", ".safetensors", ".pt", ".pth")
SECRET_RE = re.compile(rb"(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})")


def _sanitized_env(temp_root: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()}
    for key in (
        "HOME",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
        "UV_PROJECT_ENVIRONMENT",
    ):
        env.pop(key, None)
    home = temp_root / "home"
    cache = temp_root / "cache"
    config = temp_root / "config"
    home.mkdir()
    cache.mkdir()
    config.mkdir()
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_CACHE_HOME": str(cache),
            "XDG_CONFIG_HOME": str(config),
            "UV_CACHE_DIR": str(cache / "uv"),
            "PIP_CACHE_DIR": str(cache / "pip"),
            "PIP_CONFIG_FILE": os.devnull,
            "UV_NO_CONFIG": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "SOURCE_DATE_EPOCH": "0",
            "TZ": "UTC",
        }
    )
    return env


def _copy_allowlist(repo_root: Path, source_root: Path) -> None:
    for relative in SOURCE_ALLOWLIST:
        source = repo_root / relative
        destination = source_root / relative
        if source.is_symlink() or not source.exists():
            raise ValueError(f"allowlisted source is missing or symlinked: {relative}")
        if source.is_dir():
            for child in sorted(source.rglob("*")):
                if child.is_symlink():
                    raise ValueError(f"symlinked source member: {child.relative_to(repo_root)}")
                if child.is_dir() or child.name == "__pycache__" or child.suffix == ".pyc":
                    continue
                if not child.is_file():
                    raise ValueError(f"non-regular source member: {child.relative_to(repo_root)}")
                target = source_root / child.relative_to(repo_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(child, target)
        else:
            if not source.is_file():
                raise ValueError(f"non-regular allowlisted source member: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    # Setuptools does not include CHANGELOG.md in an sdist by default.  Keep
    # the Git source allowlist strict while supplying this deterministic,
    # build-only manifest so the release notes are present in the archive.
    (source_root / "MANIFEST.in").write_text("include CHANGELOG.md\nexclude MANIFEST.in\n")


def _run_build(repo_root: Path, source_root: Path, output_dir: Path, env: dict[str, str]) -> None:
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
        cwd=source_root,
        env=env,
    )


def _safe_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or PureWindowsPath(name).is_absolute()
        or ".." in path.parts
        or "\\" in name
        or "\x00" in name
    ):
        raise ValueError(f"unsafe archive member path: {name!r}")
    if name.startswith(FORBIDDEN_MEMBERS) or name.endswith(FORBIDDEN_SUFFIXES):
        raise ValueError(f"forbidden release member: {name!r}")
    _scan_payload(name, name.encode())


def _scan_payload(name: str, payload: bytes) -> None:
    for marker in FORBIDDEN_TEXT:
        if marker.encode() in payload:
            raise ValueError(f"forbidden host/secret marker {marker!r} in {name!r}")
    if SECRET_RE.search(payload):
        raise ValueError(f"high-signal secret marker in {name!r}")


def _inspect_zip(path: Path, expected: set[str] | None) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP members are forbidden")
        for info in infos:
            _safe_member_name(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            permissions = (info.external_attr >> 16) & 0o777
            if mode and mode not in (stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"symlink/special ZIP member: {info.filename!r}")
            if info.is_dir():
                if mode not in (0, stat.S_IFDIR) or permissions not in (0, 0o755):
                    raise ValueError(f"non-canonical directory mode: {info.filename!r}")
                continue
            # ZipFile may assign 0600 to synthetic/default entries; the
            # canonicalizer rewrites all regular files to 0644.
            if mode not in (0, stat.S_IFREG) or permissions not in (0, 0o600, 0o644):
                raise ValueError(f"non-canonical file mode: {info.filename!r}")
            payload = archive.read(info)
            _scan_payload(info.filename, payload)
    if expected is not None and set(names) != expected:
        raise ValueError(f"wheel members differ: {sorted(set(names) ^ expected)}")
    return names


def _inspect_tar(path: Path, expected: set[str] | None) -> list[str]:
    with (
        gzip.open(path, "rb") as compressed,
        tarfile.open(fileobj=compressed, mode="r:") as archive,
    ):
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("duplicate TAR members are forbidden")
        with tempfile.TemporaryDirectory(prefix="atlas-archive-extract-") as extract_root:
            root = Path(extract_root)
            for member in members:
                _safe_member_name(member.name)
                if not (member.isdir() or member.isfile()):
                    raise ValueError(f"symlink/hardlink/special TAR member: {member.name!r}")
                target = root / member.name
                if member.isdir():
                    if member.mode & 0o777 not in (0, 0o755):
                        raise ValueError(f"non-canonical directory mode: {member.name!r}")
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if member.mode & 0o777 not in (0, 0o644):
                    raise ValueError(f"non-canonical file mode: {member.name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                payload_file = archive.extractfile(member)
                if payload_file is None:
                    raise ValueError(f"unable to read TAR member: {member.name!r}")
                payload = payload_file.read()
                target.write_bytes(payload)
                _scan_payload(member.name, payload)
    if expected is not None and set(names) != expected:
        raise ValueError(f"sdist members differ: {sorted(set(names) ^ expected)}")
    return names


def _archive_members(path: Path, expected: set[str] | None = None) -> list[str]:
    if path.suffix == ".whl":
        return _inspect_zip(path, expected)
    return _inspect_tar(path, expected)


def _canonicalize_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as source:
        infos = source.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP members are forbidden")
        members = []
        for info in infos:
            _safe_member_name(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if mode and mode not in (stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"symlink/special ZIP member: {info.filename!r}")
            members.append((info.filename, source.read(info.filename), info.is_dir()))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data, is_dir in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if is_dir else zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o040755 if is_dir else 0o100644) << 16
            info.internal_attr = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(info, data)
    path.write_bytes(output.getvalue())


def _canonicalize_sdist(path: Path) -> None:
    with gzip.open(path, "rb") as compressed:
        source_bytes = compressed.read()
    output_tar = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(source_bytes), mode="r:") as source:
        members = source.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("duplicate TAR members are forbidden")
        for member in members:
            _safe_member_name(member.name)
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"symlink/hardlink/special TAR member: {member.name!r}")
        members = sorted(members, key=lambda member: member.name)
        with tarfile.open(fileobj=output_tar, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for original in members:
                member = copy.copy(original)
                member.mtime = 0
                member.uid = 0
                member.gid = 0
                member.uname = ""
                member.gname = ""
                member.pax_headers = {}
                if member.isdir():
                    member.mode = 0o755
                    member.size = 0
                    member.type = tarfile.DIRTYPE
                    member.linkname = ""
                    archive.addfile(member)
                elif member.isfile():
                    member.mode = 0o644
                    member.type = tarfile.REGTYPE
                    member.linkname = ""
                    data = source.extractfile(original)
                    if data is None:
                        raise ValueError(f"unable to read sdist member: {original.name}")
                    payload = data.read()
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
                else:
                    raise ValueError(f"unsupported sdist member: {original.name}")
    output = io.BytesIO()
    with gzip.GzipFile(
        fileobj=output, mode="wb", filename="", mtime=0, compresslevel=9
    ) as compressed:
        compressed.write(output_tar.getvalue())
    path.write_bytes(output.getvalue())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(paths: Iterable[Path], output: Path) -> None:
    lines = [f"{_sha256(path)}  {path.name}" for path in sorted(paths, key=lambda path: path.name)]
    output.write_text("\n".join(lines) + "\n")


def _expected_members(repo_root: Path) -> tuple[set[str], set[str]]:
    top_files = [relative for relative in SOURCE_ALLOWLIST if relative != "src/atlas_wm"]
    package_files = sorted((repo_root / "src/atlas_wm").rglob("*.py"))
    wheel_files = {str(path.relative_to(repo_root / "src")) for path in package_files}
    dist_info = f"atlas_wm-{VERSION}.dist-info"
    wheel_files.update(
        {
            f"{dist_info}/METADATA",
            f"{dist_info}/RECORD",
            f"{dist_info}/WHEEL",
            f"{dist_info}/top_level.txt",
            f"{dist_info}/licenses/LICENSE",
        }
    )
    root = f"atlas_wm-{VERSION}"
    sdist_files = {f"{root}/{relative}" for relative in top_files}
    sdist_files.update(f"{root}/{name}" for name in ("PKG-INFO", "setup.cfg"))
    sdist_files.update(f"{root}/{path.relative_to(repo_root)}" for path in package_files)
    sdist_files.update(
        f"{root}/src/atlas_wm.egg-info/{name}"
        for name in (
            "PKG-INFO",
            "SOURCES.txt",
            "dependency_links.txt",
            "requires.txt",
            "top_level.txt",
        )
    )
    sdist_dirs = {
        str(parent)
        for member in sdist_files
        for parent in PurePosixPath(member).parents
        if str(parent) != "."
    }
    return wheel_files, sdist_files | sdist_dirs


def validate_staging(repo_root: Path, output_dir: Path) -> None:
    expected_staging = {WHEEL_NAME, SDIST_NAME, "sbom.json", "SHA256SUMS"}
    entries = list(output_dir.iterdir())
    actual_staging = {path.name for path in entries}
    if actual_staging != expected_staging:
        raise ValueError(f"unexpected release staging contents: {sorted(actual_staging)}")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("release staging contains a non-regular member")
    expected_wheel, expected_sdist = _expected_members(repo_root)
    _archive_members(output_dir / WHEEL_NAME, expected_wheel)
    _archive_members(output_dir / SDIST_NAME, expected_sdist)
    _scan_payload("sbom.json", (output_dir / "sbom.json").read_bytes())
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
    if any(output_dir.iterdir()):
        raise ValueError("release staging must start empty")
    with tempfile.TemporaryDirectory(prefix="atlas-release-build-") as temp_root:
        temp_root_path = Path(temp_root)
        source_root = temp_root_path / "source"
        source_root.mkdir()
        _copy_allowlist(repo_root, source_root)
        _run_build(repo_root, source_root, output_dir, _sanitized_env(temp_root_path))
    _canonicalize_wheel(output_dir / WHEEL_NAME)
    _canonicalize_sdist(output_dir / SDIST_NAME)
    write_sbom(
        repo_root / "pyproject.toml", repo_root / "requirements.lock", output_dir / "sbom.json"
    )
    _write_checksums(
        [output_dir / WHEEL_NAME, output_dir / SDIST_NAME, output_dir / "sbom.json"],
        output_dir / "SHA256SUMS",
    )
    validate_staging(repo_root, output_dir)
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
