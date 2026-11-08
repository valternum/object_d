"""Preparation of YAGO 4.6 downloads for ``holm-index build`` (stdlib only).

Archives are treated as untrusted data (R-D3): members are validated before
anything is written, nothing is executed, and checksums are recorded (R-D1).

Memory: O(1) in the file sizes (streaming). Time: O(bytes).
"""

import hashlib
import shutil
import zipfile
from pathlib import Path

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(_CHUNK):
            h.update(block)
    return h.hexdigest()


def _is_data_line(line: str) -> bool:
    """A TAB-separated one-line triple ``s<TAB>p<TAB>o<TAB>.`` (YAGO data format)."""
    return (
        not line[:1].isspace()
        and not line.startswith("@prefix")
        and line.count("\t") >= 3
        and line.rstrip().endswith(".")
    )


def split_tiny(src: Path, schema_out: Path, data_out: Path) -> tuple[int, int]:
    """Split ``yago-tiny.ttl`` into its leading multi-line schema block and the data.

    ``@prefix`` lines are copied into both outputs. Returns the number of
    non-prefix lines written to ``(schema_out, data_out)``.
    """
    prefixes: list[str] = []
    n_schema = n_data = 0
    in_data = False
    with (
        src.open(encoding="utf-8", newline="") as fin,
        schema_out.open("w", encoding="utf-8", newline="") as fs,
        data_out.open("w", encoding="utf-8", newline="") as fd,
    ):
        for line in fin:
            if not in_data:
                if line.startswith("@prefix"):
                    prefixes.append(line)
                    fs.write(line)
                    continue
                if _is_data_line(line):
                    in_data = True
                    fd.writelines(prefixes)
                else:
                    fs.write(line)
                    n_schema += 1
                    continue
            fd.write(line)
            n_data += 1
    if not in_data:
        raise ValueError(f"{src}: no data line found")
    return n_schema, n_data


def extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    """Extract all members into ``out_dir``; refuse members escaping it."""
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as z:
        members = z.infolist()
        for info in members:
            target = (root / info.filename).resolve()
            if info.filename.startswith(("/", "\\")) or root not in target.parents:
                raise ValueError(f"{zip_path}: unsafe member {info.filename!r}")
        out: list[Path] = []
        for info in members:
            if info.is_dir():
                continue
            dest = out_dir / info.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as fsrc, dest.open("wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, _CHUNK)
            out.append(dest)
    return out
