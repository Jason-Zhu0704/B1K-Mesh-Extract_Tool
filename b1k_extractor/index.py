"""Batch extraction and asset index management."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# ── asset discovery ──────────────────────────────────────────────────────────

def iter_assets(
    assets_root: Path,
    category: str | None = None,
    model: str | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield (category, model) pairs for B1K assets under *assets_root*.

    Args:
        assets_root: path to ``behavior-1k-assets/objects/``
        category: optional filter (exact name)
        model: optional filter (exact name; requires category)
    """
    cats = sorted(assets_root.iterdir()) if assets_root.exists() else []
    if category:
        cats = [p for p in cats if p.name == category]

    for cat_dir in cats:
        if not cat_dir.is_dir():
            continue
        models = sorted(cat_dir.iterdir())
        if model:
            models = [p for p in models if p.name == model]
        for model_dir in models:
            if not model_dir.is_dir():
                continue
            usd_dir = model_dir / "usd"
            if usd_dir.exists():
                yield cat_dir.name, model_dir.name


def count_assets(assets_root: Path, category: str | None = None) -> int:
    return sum(1 for _ in iter_assets(assets_root, category=category))


# ── metadata (fast, no OmniGibson) ──────────────────────────────────────────

def load_metadata(assets_root: Path, category: str, model: str) -> dict | None:
    """Load ``misc/metadata.json`` for one asset."""
    p = assets_root / category / model / "misc" / "metadata.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Cannot read metadata for {category}/{model}: {e}")
        return None


def bbox_from_metadata(meta: dict) -> list[float] | None:
    """Extract [x, y, z] bounding-box size in metres from metadata."""
    return meta.get("bbox_size")


# ── index management ─────────────────────────────────────────────────────────

def load_index(index_path: Path) -> dict:
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return {}


def save_index(index: dict, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info(f"Index saved → {index_path}  ({len(index)} entries)")


def update_index_entry(
    index: dict,
    category: str,
    model: str,
    written_files: dict[str, Path],
    metadata: dict | None = None,
) -> None:
    """Update the index dict in-place for one asset."""
    key = f"{category}-{model}"
    entry = index.setdefault(key, {"category": category, "model": model})

    if metadata is not None:
        entry["bbox_size"] = metadata.get("bbox_size")
        entry["base_link_offset"] = metadata.get("base_link_offset")

    for role_fmt, path in written_files.items():
        # e.g. "visual.obj" → key "visual_obj"
        role, fmt = role_fmt.split(".", 1)
        field = f"{role}_{fmt}"
        entry[field] = str(path)
