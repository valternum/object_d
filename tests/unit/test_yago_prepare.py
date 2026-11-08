"""YAGO download preparation: tiny splitting, safe extraction, checksums (R-D1, R-D3)."""

import zipfile
from pathlib import Path

import pytest

from holm.data.yago_prepare import extract_zip, sha256_file, split_tiny

TINY = (
    "@prefix yago: <http://yago-knowledge.org/resource/> .\n"
    "@prefix schema: <http://schema.org/> .\n"
    "\n"
    "schema:Thing a sh:NodeShape ;\n"
    '\trdfs:label "Thing"@en ;\n'
    "\ta rdfs:Class .\n"
    "\n"
    "yago:A\trdf:type\tschema:Thing\t.\n"
    'yago:B rdfs:label "b"@en; rdf:type schema:Thing .\n'
    "yago:A\tschema:knows\tyago:B\t.\n"
)


def test_split_tiny_separates_schema_block_from_data(tmp_path: Path) -> None:
    src = tmp_path / "yago-tiny.ttl"
    src.write_text(TINY, encoding="utf-8")
    schema, data = tmp_path / "schema.ttl", tmp_path / "data.ttl"
    n_schema, n_data = split_tiny(src, schema, data)
    assert (n_schema, n_data) == (5, 3)
    schema_text = schema.read_text(encoding="utf-8")
    data_text = data.read_text(encoding="utf-8")
    assert "schema:Thing a sh:NodeShape ;" in schema_text
    assert "yago:A\trdf:type" not in schema_text
    # Prefixes are copied into both parts so each file is valid on its own.
    assert data_text.startswith("@prefix yago:")
    assert data_text.count("\n") == 2 + 3
    assert data_text.rstrip().endswith("yago:A\tschema:knows\tyago:B\t.")


def test_split_tiny_without_data_raises(tmp_path: Path) -> None:
    src = tmp_path / "t.ttl"
    src.write_text("@prefix a: <x> .\nschema:T a sh:NodeShape .\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data line"):
        split_tiny(src, tmp_path / "s.ttl", tmp_path / "d.ttl")


def test_sha256_known_vector(tmp_path: Path) -> None:
    p = tmp_path / "abc"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_extract_zip_writes_members(tmp_path: Path) -> None:
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("yago-schema.ttl", "x")
    out = tmp_path / "out"
    assert extract_zip(z, out) == [out / "yago-schema.ttl"]
    assert (out / "yago-schema.ttl").read_text() == "x"


def test_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("../evil.txt", "boom")
    with pytest.raises(ValueError, match="unsafe member"):
        extract_zip(z, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()
