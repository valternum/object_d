# HOLM build and test entry points (see .claude/TESTING.md).
# Works with GNU make from Git Bash on Windows; cargo must be on PATH.

PY ?= $(CURDIR)/.venv/Scripts/python.exe
CARGO ?= cargo
RUST_MANIFEST := rust/Cargo.toml
PYTEST_FAST := not contract and not integration and not smoke and not resource

export PYO3_PYTHON := $(PY)
export VIRTUAL_ENV := $(CURDIR)/.venv

.PHONY: build-native test-fast test test-gpu lint fmt

build-native:
	$(PY) -m maturin develop --release

test-fast:
	$(CARGO) test --manifest-path $(RUST_MANIFEST) -q
	$(PY) -m pytest -q -m "$(PYTEST_FAST)"

test:
	$(CARGO) test --manifest-path $(RUST_MANIFEST) -q
	$(PY) -m pytest -q -m "not smoke and not resource"

test-gpu:
	$(PY) -m pytest -q -m "smoke or resource"

lint:
	$(CARGO) fmt --manifest-path $(RUST_MANIFEST) --all -- --check
	$(CARGO) clippy --manifest-path $(RUST_MANIFEST) --workspace --all-targets -- -D warnings
	$(PY) -m ruff check python tests scripts
	$(PY) -m ruff format --check python tests scripts
	$(PY) -m mypy --strict python/holm/retrieval python/holm/reference

fmt:
	$(CARGO) fmt --manifest-path $(RUST_MANIFEST) --all
	$(PY) -m ruff format python tests scripts
