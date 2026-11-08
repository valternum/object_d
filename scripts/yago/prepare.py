"""Prepare downloaded YAGO 4.6 archives for ``holm-index build``.

Usage (run with ``python -I``; archives are untrusted data, R-D3)::

    python -I scripts/yago/prepare.py --src DIR_WITH_ZIPS --out DIR

Extracts every ``*.zip`` of ``--src`` into ``--out``, writes ``sources.json``
(file name, size, sha256 of each archive; R-D1) and, if ``yago-tiny.ttl`` is
present, splits it into ``tiny-schema.ttl`` and ``tiny-data.ttl``.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from holm.data.yago_prepare import extract_zip, sha256_file, split_tiny  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = []
    for z in sorted(args.src.glob("*.zip")):
        sources.append({"archive": z.name, "bytes": z.stat().st_size, "sha256": sha256_file(z)})
        for p in extract_zip(z, args.out):
            print(f"extracted {p}")
    (args.out / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    tiny = args.out / "yago-tiny.ttl"
    if tiny.exists():
        n_schema, n_data = split_tiny(tiny, args.out / "tiny-schema.ttl", args.out / "tiny-data.ttl")
        print(f"tiny: {n_schema} schema lines, {n_data} data lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
