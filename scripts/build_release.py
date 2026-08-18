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
import stat
import subprocess
import sys
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
SOURCE_TOP_LEVEL = frozenset(SOURCE_ALLOWLIST[:-1])
CONTROL_FILES = (
    "pyproject.toml",
    "requirements.lock",
    "scripts/build_release.py",
    "scripts/generate_sbom.py",
    "uv.lock",
)
EXPECTED_PYTHON = (3, 11, 15)
ARTIFACT_NAMES = (WHEEL_NAME, SDIST_NAME, "sbom.json")
STAGING_NAMES = frozenset((*ARTIFACT_NAMES, "SHA256SUMS"))
PASSTHROUGH_ENV = frozenset(
    {
        "PATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
    }
)
FORBIDDEN_TEXT = (
    "file://",
    "/Users/",
    "/home/runner",
    "/workspace",
    "/workspaces",
    "/private/" + "tmp/",
    "/" + "tmp/",
    "/var/" + "folders/",
    "-----BEGIN ",
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
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
SECRET_RE = re.compile(
    rb"(?:"
    rb"(?:AKIA|ASIA|AIDA|AROA|AGPA|A3T)[0-9A-Z]{16,17}|"
    rb"github_pat_[A-Za-z0-9_]{20,}|"
    rb"gh[opusr]_[A-Za-z0-9_]{20,}|"
    rb"glpat-[A-Za-z0-9_-]{20,}|"
    rb"npm_[A-Za-z0-9]{20,}|"
    rb"xox[baprs]-[A-Za-z0-9-]{10,}|"
    rb"(?:sk|xai)-[A-Za-z0-9_-]{20,}"
    rb")"
)


def _assert_python() -> None:
    if sys.version_info[:3] != EXPECTED_PYTHON:
        actual = ".".join(str(part) for part in sys.version_info[:3])
        expected = ".".join(str(part) for part in EXPECTED_PYTHON)
        raise RuntimeError(f"release build requires Python {expected}, got {actual}")


def _git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    # GIT_DIR/GIT_WORK_TREE and user/system config can redirect all Git
    # authority away from repo_root. Keep only path/locale settings needed by
    # Git and explicitly disable ambient config and prompts.
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE"}
    }
    env.setdefault("PATH", os.defpath)
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return subprocess.run(
        ["git", "--no-replace-objects", "--no-pager", "-C", str(repo_root), *args],
        check=check,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _head_blob(repo_root: Path, relative: str) -> bytes:
    return _git(repo_root, "show", f"HEAD:{relative}").stdout


def _assert_git_authority(repo_root: Path) -> None:
    expected_root = repo_root.resolve()
    actual_root = Path(
        _git(repo_root, "rev-parse", "--show-toplevel").stdout.decode().strip()
    ).resolve()
    if actual_root != expected_root:
        raise ValueError(f"Git repository mismatch: expected {expected_root}, got {actual_root}")
    head = _git(repo_root, "rev-parse", "--verify", "HEAD^{commit}").stdout.decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError(f"Git HEAD is not a full commit: {head!r}")


def _assert_control_files(repo_root: Path) -> None:
    for relative in CONTROL_FILES:
        path = repo_root / relative
        if not path.is_file() or path.read_bytes() != _head_blob(repo_root, relative):
            raise ValueError(f"control file is not identical to HEAD: {relative}")


def _tracked_source_files(repo_root: Path) -> list[str]:
    result = _git(
        repo_root,
        "ls-tree",
        "-r",
        "-z",
        "HEAD",
        "--",
        *SOURCE_ALLOWLIST,
    )
    files: list[str] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, _object_id = metadata.decode("ascii").split()
        relative = raw_path.decode("utf-8")
        if kind != "blob" or mode != "100644":
            raise ValueError(f"allowlisted Git member is not a regular file: {relative}")
        if relative in SOURCE_TOP_LEVEL or relative.startswith("src/atlas_wm/"):
            files.append(relative)
    files.sort()
    if set(files) & SOURCE_TOP_LEVEL != SOURCE_TOP_LEVEL:
        missing = sorted(SOURCE_TOP_LEVEL - set(files))
        raise ValueError(f"required Git release source is missing: {missing}")
    if not any(relative.startswith("src/atlas_wm/") for relative in files):
        raise ValueError("Git release source has no atlas_wm package files")
    return files


def _assert_clean_tree(repo_root: Path) -> None:
    result = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if result.stdout:
        status = result.stdout.decode("utf-8", errors="replace").strip()
        raise ValueError(f"release build requires a clean Git tree:\n{status}")


def _sanitized_env(temp_root: Path) -> dict[str, str]:
    # Do not inherit credentials, package indexes, CI tokens, or arbitrary
    # config. PATH is needed to resolve uv; CA and locale variables are the
    # only other host settings that can be required for a package download.
    env = {key: value for key, value in os.environ.items() if key in PASSTHROUGH_ENV}
    env.setdefault("PATH", os.defpath)
    home = temp_root / "home"
    cache = temp_root / "cache"
    config = temp_root / "config"
    tmp = temp_root / "tmp"
    home.mkdir()
    cache.mkdir()
    config.mkdir()
    tmp.mkdir()
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "TMPDIR": str(tmp),
            "TMP": str(tmp),
            "TEMP": str(tmp),
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


def _copy_allowlist(repo_root: Path, source_root: Path, tracked_files: list[str]) -> None:
    for relative in tracked_files:
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_head_blob(repo_root, relative))
    # Setuptools does not include CHANGELOG.md in an sdist by default.  Keep
    # the Git source allowlist strict while supplying this deterministic,
    # build-only manifest so the release notes are present in the archive.
    (source_root / "MANIFEST.in").write_text("include CHANGELOG.md\nexclude MANIFEST.in\n")


def _run_build(
    source_root: Path, output_dir: Path, build_constraints: Path, env: dict[str, str]
) -> None:
    _assert_python()
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
            "--python",
            sys.executable,
            "--build-constraints",
            str(build_constraints),
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


def _canonical_payload(name: str, payload: bytes) -> bytes:
    """Normalize text payloads whose backend may vary line endings/order."""
    if name.endswith((".cfg", ".md", ".py", ".pyproject", ".txt", ".toml")):
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if name.endswith("/SOURCES.txt"):
        payload = b"\n".join(sorted(payload.splitlines())) + b"\n"
    return payload


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
            members.append(
                (
                    info.filename,
                    _canonical_payload(info.filename, source.read(info.filename)),
                    info.is_dir(),
                )
            )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data, is_dir in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if is_dir else zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.create_version = 20
            info.extract_version = 20
            info.flag_bits = 0
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
                    payload = _canonical_payload(original.name, data.read())
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


def _staging_hashes(output_dir: Path) -> dict[str, str]:
    entries = list(output_dir.iterdir())
    if {path.name for path in entries} != STAGING_NAMES or any(
        path.is_symlink() or not path.is_file() for path in entries
    ):
        raise ValueError("staging contains files outside the release allowlist")
    pattern = re.compile(r"^(?P<digest>[0-9a-f]{64})  (?P<name>[^\s]+)$")
    checksums: dict[str, str] = {}
    for line in (output_dir / "SHA256SUMS").read_text().splitlines():
        match = pattern.fullmatch(line)
        if match is None or match.group("name") not in ARTIFACT_NAMES:
            raise ValueError(f"invalid SHA256SUMS line: {line!r}")
        name = match.group("name")
        if name in checksums:
            raise ValueError(f"duplicate SHA256SUMS member: {name}")
        checksums[name] = match.group("digest")
    if set(checksums) != set(ARTIFACT_NAMES):
        raise ValueError("SHA256SUMS does not enumerate exactly the release artifacts")
    computed = {name: _sha256(output_dir / name) for name in ARTIFACT_NAMES}
    if checksums != computed:
        raise ValueError("SHA256SUMS does not match downloaded artifact bytes")
    return {**computed, "SHA256SUMS": _sha256(output_dir / "SHA256SUMS")}


def compare_staging(repo_root: Path, first: Path, second: Path) -> dict[str, str]:
    """Validate and compare two downloaded platform staging directories."""
    validate_staging(repo_root, first)
    validate_staging(repo_root, second)
    first_hashes = _staging_hashes(first)
    second_hashes = _staging_hashes(second)
    if first_hashes != second_hashes:
        raise ValueError(
            f"cross-platform artifact hashes differ: {first_hashes} != {second_hashes}"
        )
    for name in (*ARTIFACT_NAMES, "SHA256SUMS"):
        if (first / name).read_bytes() != (second / name).read_bytes():
            raise ValueError(f"cross-platform artifact bytes differ: {name}")
    return first_hashes


def _expected_members(repo_root: Path) -> tuple[set[str], set[str]]:
    tracked_files = _tracked_source_files(repo_root)
    top_files = sorted(SOURCE_TOP_LEVEL)
    package_files = [relative for relative in tracked_files if relative.startswith("src/atlas_wm/")]
    wheel_files = {relative.removeprefix("src/") for relative in package_files}
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
    sdist_files.update(f"{root}/{path}" for path in package_files)
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
    _staging_hashes_before = _staging_hashes(output_dir)
    expected_wheel, expected_sdist = _expected_members(repo_root)
    _archive_members(output_dir / WHEEL_NAME, expected_wheel)
    _archive_members(output_dir / SDIST_NAME, expected_sdist)
    _scan_payload("sbom.json", (output_dir / "sbom.json").read_bytes())
    if _staging_hashes(output_dir) != _staging_hashes_before:
        raise ValueError("staging changed while being validated")


def build_artifacts(repo_root: Path, output_dir: Path) -> dict[str, str]:
    """Build one clean staging directory and return artifact hashes."""
    _assert_python()
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    _assert_git_authority(repo_root)
    _assert_clean_tree(repo_root)
    _assert_control_files(repo_root)
    tracked_files = _tracked_source_files(repo_root)
    if output_dir == repo_root or repo_root in output_dir.parents:
        raise ValueError("release staging must be outside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("release staging must start empty")
    with tempfile.TemporaryDirectory(prefix="atlas-release-build-") as temp_root:
        temp_root_path = Path(temp_root)
        source_root = temp_root_path / "source"
        controls_root = temp_root_path / "controls"
        source_root.mkdir()
        controls_root.mkdir()
        _copy_allowlist(repo_root, source_root, tracked_files)
        for relative in ("pyproject.toml", "requirements.lock"):
            control_path = controls_root / relative
            control_path.parent.mkdir(parents=True, exist_ok=True)
            control_path.write_bytes(_head_blob(repo_root, relative))
        _run_build(
            source_root,
            output_dir,
            controls_root / "requirements.lock",
            _sanitized_env(temp_root_path),
        )
        write_sbom(
            controls_root / "pyproject.toml",
            controls_root / "requirements.lock",
            output_dir / "sbom.json",
        )
    _canonicalize_wheel(output_dir / WHEEL_NAME)
    _canonicalize_sdist(output_dir / SDIST_NAME)
    _write_checksums(
        [output_dir / WHEEL_NAME, output_dir / SDIST_NAME, output_dir / "sbom.json"],
        output_dir / "SHA256SUMS",
    )
    validate_staging(repo_root, output_dir)
    return {path.name: _sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if (args.output_dir is None) == (args.validate_dir is None):
        parser.error("exactly one of --output-dir or --validate-dir is required")
    repo_root = args.repo_root.resolve()
    _assert_python()
    _assert_git_authority(repo_root)
    _assert_clean_tree(repo_root)
    _assert_control_files(repo_root)
    if args.validate_dir is not None:
        validate_staging(repo_root, args.validate_dir.resolve())
        hashes = _staging_hashes(args.validate_dir.resolve())
    else:
        hashes = build_artifacts(repo_root, args.output_dir)
    for name, digest in hashes.items():
        print(f"{digest}  {name}")


if __name__ == "__main__":
    main()
