#!/usr/bin/env python3
"""b1k-extract — Command-line tool to convert B1K encrypted USD assets to mesh files.

Requires OmniGibson + Isaac Sim (for USD decryption).

Examples
--------
# Extract one asset (OBJ only)
b1k-extract --category bed --model fnzxxr --output ./meshes

# Extract all beds, output OBJ + GLB + STL
b1k-extract --category bed --output ./meshes --formats obj glb stl

# Extract everything (slow — one OmniGibson load per asset)
b1k-extract --output ./meshes

# List available categories
b1k-extract --list-categories
"""

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _default_assets_root() -> Path:
    candidates = [
        Path("/data/BEHAVIOR-1K/datasets/behavior-1k-assets/objects"),
        Path.home() / "BEHAVIOR-1K/datasets/behavior-1k-assets/objects",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def cmd_list_categories(assets_root: Path) -> None:
    from b1k_extractor.index import iter_assets
    cats = sorted({cat for cat, _ in iter_assets(assets_root)})
    for cat in cats:
        n = sum(1 for c, _ in iter_assets(assets_root, category=cat))
        print(f"  {cat:<40} ({n} models)")
    print(f"\nTotal: {len(cats)} categories")


def cmd_extract(args: argparse.Namespace) -> int:
    import time
    from b1k_extractor.index import (
        iter_assets,
        load_metadata,
        load_index,
        save_index,
        update_index_entry,
    )
    from b1k_extractor.extractor import AssetExtractor

    log = logging.getLogger("b1k_extract")
    assets_root = Path(args.assets_root)
    output_dir = Path(args.output)
    index_path = output_dir / "index.json"
    formats = args.formats or ["obj"]

    # ── Collect targets ──────────────────────────────────────────────
    all_targets = list(iter_assets(
        assets_root,
        category=args.category or None,
        model=args.model or None,
    ))
    if not all_targets:
        log.error("No assets found matching the given filters.")
        return 1

    index = load_index(index_path)

    # Filter out already-done unless --force
    if args.force:
        pending = all_targets
    else:
        pending = []
        for cat, model in all_targets:
            key = f"{cat}-{model}"
            existing = index.get(key, {})
            if any(f"visual_{fmt}" in existing for fmt in formats):
                continue
            pending.append((cat, model))

    skipped = len(all_targets) - len(pending)
    log.info(f"Total: {len(all_targets)} assets  |  to extract: {len(pending)}  |  skipped (done): {skipped}")
    log.info(f"Output: {output_dir}  |  formats: {formats}  |  batch_size: {args.batch_size}")

    if not pending:
        log.info("All assets already extracted.")
        return 0

    # ── Batch extraction ──────────────────────────────────────────────
    errors: list[str] = []
    done = 0
    t_start = time.time()

    for result in AssetExtractor.extract_batch(pending, batch_size=args.batch_size):
        cat, model = result.category, result.model
        if result.visual is None and result.collision_hull is None:
            errors.append(f"{cat}/{model}: no geometry")
            continue
        meta = load_metadata(assets_root, cat, model)
        out = output_dir / "objects" / cat / model
        written = result.save(out, formats=formats)
        update_index_entry(index, cat, model, written, metadata=meta)
        done += 1
        if done % 10 == 0:
            save_index(index, index_path)
            elapsed = time.time() - t_start
            rate = done / elapsed
            remaining = (len(pending) - done) / rate if rate > 0 else 0
            log.info(
                f"  [{done}/{len(pending)}]  "
                f"{rate:.2f} assets/s  "
                f"ETA {remaining/3600:.1f}h"
            )

    save_index(index, index_path)

    if errors:
        log.warning(f"{len(errors)} errors:")
        for err in errors:
            log.warning(f"  {err}")

    elapsed = time.time() - t_start
    log.info(f"Done. {done} extracted in {elapsed/3600:.2f}h  ({len(errors)} errors)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="b1k-extract",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--assets-root",
        default=str(_default_assets_root()),
        metavar="DIR",
        help="Path to behavior-1k-assets/objects/ (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output", "-o",
        default="b1k_meshes",
        metavar="DIR",
        help="Output directory (default: b1k_meshes/)",
    )
    parser.add_argument(
        "--category", "-c",
        default=None,
        help="Limit to one category, e.g. 'bed'",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Limit to one model ID (requires --category)",
    )
    parser.add_argument(
        "--formats", "-f",
        nargs="+",
        default=["obj"],
        metavar="FMT",
        help="Output formats: obj glb stl ply (default: obj)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        metavar="N",
        help="Assets per OmniGibson session (default: 20, higher = faster but more RAM)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract assets even if already in index",
    )
        action="store_true",
        help="Print available categories and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.model and not args.category:
        parser.error("--model requires --category")

    if args.list_categories:
        cmd_list_categories(Path(args.assets_root))
        return

    sys.exit(cmd_extract(args))


if __name__ == "__main__":
    main()
