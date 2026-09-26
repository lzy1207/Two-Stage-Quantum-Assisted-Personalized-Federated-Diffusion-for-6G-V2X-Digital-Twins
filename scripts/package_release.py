#!/usr/bin/env python3
"""Build and verify a portable source release using only the standard library.

The archive contains one top-level project directory and an in-archive
MANIFEST.sha256 covering every packaged source/example file (not itself).
Runs, virtual environments, caches, VCS data, and compiled Python files are
excluded. Existing output files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile


ALLOWED_DIRECTORIES = {"qv2x", "src", "configs", "config", "docs", "tests", "scripts", "examples"}
ALLOWED_ROOT_FILES = {
    "pyproject.toml", "setup.py", "setup.cfg", ".gitignore", ".gitattributes",
    "run_demo.ps1", "run_demo.sh", "environment.yml", "environment.yaml",
    "requirements.txt", "requirements-dev.txt", "requirements-test.txt",
    "requirements-cpu.txt", "requirements-cuda.txt", "MANIFEST.in",
}
EXCLUDED_DIRECTORIES = {
    "runs", ".venv", "venv", "env", ".git", ".hg", ".svn", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox",
    "node_modules", "build", "dist", ".idea", ".vscode",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".swp", ".swo", ".zip", ".rar", ".7z"}
MANIFEST_NAME = "MANIFEST.sha256"
CHUNK_SIZE = 1024 * 1024


def source_files(project_root: Path) -> list[Path]:
    """Whitelist release content; reject symlinks instead of following them."""
    root = project_root.resolve()
    files = []
    # Only recurse whitelisted roots: traversing runs/.venv would be wasteful.
    for item in root.iterdir():
        if item.is_symlink():
            continue
        if item.is_dir() and item.name in ALLOWED_DIRECTORIES:
            candidates = item.rglob("*")
        elif item.is_file() and (item.name in ALLOWED_ROOT_FILES
                                or item.name.lower().startswith(("readme", "license", "notice", "citation"))
                                or item.name.startswith("requirements") and item.suffix == ".txt"):
            candidates = (item,)
        else:
            continue
        for path in candidates:
            relative = path.relative_to(root)
            if any(part.lower() in EXCLUDED_DIRECTORIES or part.lower().endswith(".egg-info")
                   for part in relative.parts):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES or path.name in {MANIFEST_NAME, ".DS_Store", "Thumbs.db"}:
                continue
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Source escapes project directory: {relative}")
            if "\n" in relative.as_posix() or "\r" in relative.as_posix():
                raise ValueError("Manifest filenames cannot contain newlines")
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _zip_info(name: str) -> zipfile.ZipInfo:
    # Fixed timestamps make archives reproducible for identical source contents.
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
    return info


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, top_directory: str) -> dict:
    """Check ZIP CRC and every decompressed entry against its SHA-256 manifest."""
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate entry names")
        bad_entry = archive.testzip()
        if bad_entry is not None:
            raise ValueError(f"ZIP CRC verification failed: {bad_entry}")
        manifest_entry = f"{top_directory}/{MANIFEST_NAME}"
        manifest = archive.read(manifest_entry).decode("utf-8")
        expected = {}
        for line in manifest.splitlines():
            digest, relative = line.split("  ", 1)
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("Malformed manifest SHA-256")
            if relative.startswith("/") or ".." in Path(relative).parts:
                raise ValueError("Unsafe manifest path")
            entry = f"{top_directory}/{relative}"
            if entry in expected:
                raise ValueError("Duplicate manifest entry")
            expected[entry] = digest
        if set(names) != set(expected) | {manifest_entry}:
            raise ValueError("Manifest does not cover exactly all source entries")
        total_bytes = 0
        for entry, expected_hash in expected.items():
            digest = hashlib.sha256()
            with archive.open(entry, "r") as handle:
                for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
                    total_bytes += len(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f"Manifest SHA-256 verification failed: {entry}")
    return {"source_file_count": len(expected), "archive_entry_count": len(names),
            "uncompressed_source_bytes": total_bytes, "crc_verified": True,
            "manifest_verified": True}


def package_release(project_root: str | Path, output: str | Path | None = None) -> dict:
    """Create a verified release. Existing source and output files are untouched."""
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project directory does not exist: {root}")
    target = Path(output).resolve() if output else root.parent / f"{root.name}.zip"
    if target.exists():
        raise FileExistsError(f"Output already exists; choose another --output path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    files = source_files(root)
    if not files:
        raise ValueError("No release source files found")
    # Keep all temporary writes in a new directory owned by this invocation.
    with tempfile.TemporaryDirectory(prefix=".release-build-", dir=target.parent) as temporary_directory:
        staged = Path(temporary_directory) / "release.zip"
        manifest_lines = []
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9,
                             allowZip64=True) as archive:
            for path in files:
                relative = path.relative_to(root).as_posix()
                digest = hashlib.sha256()
                before = path.stat()
                with path.open("rb") as source, archive.open(_zip_info(f"{root.name}/{relative}"), "w",
                                                            force_zip64=True) as destination:
                    for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                        digest.update(chunk)
                        destination.write(chunk)
                after = path.stat()
                if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                    raise RuntimeError(f"Source changed while packaging; retry when edits finish: {relative}")
                manifest_lines.append(f"{digest.hexdigest()}  {relative}\n")
            archive.writestr(_zip_info(f"{root.name}/{MANIFEST_NAME}"), "".join(manifest_lines).encode("utf-8"))
        report = verify_archive(staged, root.name)
        # Exclusive create also prevents overwriting a file created concurrently.
        with staged.open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination, CHUNK_SIZE)
    # Re-read the delivered file as well, so reported verification covers it.
    report = verify_archive(target, root.name)
    report.update({"output": str(target), "zip_bytes": target.stat().st_size,
                   "zip_sha256": _file_hash(target)})
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="ZIP destination; default is parent/<project-name>.zip")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    report = package_release(root, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
