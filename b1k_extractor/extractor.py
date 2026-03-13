"""Core extraction logic: encrypted B1K USD → body-local OBJ meshes.

The key insight is that B1K USD assets are encrypted for use with
OmniGibson/Isaac Sim only, but once loaded into memory, the geometry
data is fully accessible via the pxr (OpenUSD) Python API.

Coordinate convention:
    All exported meshes are in **body-local space** — i.e., relative to
    the object root prim.  This matches what Drake expects when the mesh
    is registered with an identity RigidTransform on a free body.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _triangulate(face_indices: np.ndarray, face_counts: np.ndarray) -> np.ndarray:
    """Fan-triangulate polygonal faces → (N, 3) int array."""
    tris: list[list[int]] = []
    idx = 0
    for count in face_counts:
        fan = face_indices[idx: idx + count]
        for k in range(1, count - 1):
            tris.append([fan[0], fan[k], fan[k + 1]])
        idx += count
    return np.array(tris, dtype=np.int32) if tris else np.empty((0, 3), dtype=np.int32)


def _body_local_transform(
    xform_cache,
    root_prim,
    prim,
    root_world_inv: np.ndarray,
) -> np.ndarray:
    """Return 4×4 transform: prim geometry → body-local (object root) space."""
    prim_world = np.array(xform_cache.GetLocalToWorldTransform(prim)).reshape(4, 4)
    return root_world_inv @ prim_world


# ── per-asset extraction ─────────────────────────────────────────────────────

class AssetExtractor:
    """Extract visual and collision meshes from B1K assets via OmniGibson.

    Two usage patterns:

    **Single asset** (simple, ~30 s/asset — OmniGibson starts fresh each time)::

        result = AssetExtractor.extract_one("bed", "fnzxxr")
        result.save(Path("output/bed/fnzxxr"))

    **Batch** (fast, ~3 s/asset — one OmniGibson session per batch of N)::

        results = AssetExtractor.extract_batch(
            [("bed","fnzxxr"), ("sofa","ahgkci"), ...],
            batch_size=20,
        )
        for r in results:
            r.save(Path(f"output/{r.category}/{r.model}"))
    """

    # ------------------------------------------------------------------
    @staticmethod
    def extract_one(category: str, model: str) -> "ExtractionResult":
        """Spawn a fresh OmniGibson env, extract one asset, close."""
        results = list(AssetExtractor.extract_batch([(category, model)], batch_size=1))
        return results[0]

    # ------------------------------------------------------------------
    @staticmethod
    def extract_batch(
        assets: list[tuple[str, str]],
        batch_size: int = 20,
    ):
        """Yield ExtractionResult for each (category, model) pair.

        Loads *batch_size* assets per OmniGibson session to amortise the
        ~25 s startup cost.  Each batch opens a fresh env so there is no
        USD stage accumulation across batches.
        """
        import omnigibson as og
        from omnigibson.macros import gm

        for batch_start in range(0, len(assets), batch_size):
            batch = assets[batch_start: batch_start + batch_size]

            # Build cfg with all assets in this batch
            objects_cfg = []
            for cat, model in batch:
                objects_cfg.append({
                    "type": "DatasetObject",
                    "name": f"x{cat}_{model}",
                    "category": cat,
                    "model": model,
                    "position": [0, 0, 0],
                    "visual_only": True,
                })

            # Use gm.unlocked() so macros can be set on every batch restart
            with gm.unlocked():
                gm.HEADLESS = True
                gm.USE_GPU_DYNAMICS = False
                gm.ENABLE_FLATCACHE = False
                gm.ENABLE_OBJECT_STATES = True
                gm.RENDER_VIEWER_CAMERA = False

            env = og.Environment(
                configs={"scene": {"type": "Scene"}, "objects": objects_cfg}
            )
            og.sim.step()

            try:
                for cat, model in batch:
                    obj_name = f"x{cat}_{model}"
                    obj = env.scene.object_registry("name", obj_name)
                    if obj is None:
                        logger.warning(f"  object not found in registry: {obj_name}")
                        yield ExtractionResult(cat, model, None, None)
                        continue
                    try:
                        yield AssetExtractor._extract_from_stage(obj, cat, model)
                    except Exception as e:
                        logger.error(f"  extraction failed for {cat}/{model}: {e}")
                        yield ExtractionResult(cat, model, None, None)
            finally:
                env.close()

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_from_stage(obj, category: str, model: str) -> "ExtractionResult":
        """Read geometry from the live USD stage for the given object prim."""
        import omnigibson as og
        from pxr import Usd, UsdGeom
        import trimesh

        logger.info(f"  extracting {category}/{model} …")

        stage = og.sim.stage
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        root_prim = stage.GetPrimAtPath(obj.prim_path)

        root_world = np.array(
            xform_cache.GetLocalToWorldTransform(root_prim)
        ).reshape(4, 4)
        root_world_inv = np.linalg.inv(root_world)

        visual_meshes: list = []
        collision_meshes: list = []

        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue

            prim_path_str = str(prim.GetPath()).lower()
            is_collision = "collision" in prim_path_str

            if not is_collision:
                img = UsdGeom.Imageable(prim)
                vis_attr = img.GetVisibilityAttr()
                if vis_attr.HasValue() and vis_attr.Get() == UsdGeom.Tokens.invisible:
                    continue

            mg = UsdGeom.Mesh(prim)
            pts_attr = mg.GetPointsAttr()
            if not pts_attr.HasValue():
                continue
            pts = pts_attr.Get()
            if pts is None or len(pts) == 0:
                continue

            fi_attr = mg.GetFaceVertexIndicesAttr()
            fc_attr = mg.GetFaceVertexCountsAttr()
            if not fi_attr.HasValue() or not fc_attr.HasValue():
                continue

            pts = np.array(pts, dtype=np.float64)
            fi = np.array(fi_attr.Get())
            fc = np.array(fc_attr.Get())
            faces = _triangulate(fi, fc)
            if len(faces) == 0:
                continue

            local2body = _body_local_transform(
                xform_cache, root_prim, prim, root_world_inv
            )
            tm = trimesh.Trimesh(vertices=pts, faces=faces, process=False)
            tm.apply_transform(local2body)

            if is_collision:
                collision_meshes.append(tm)
            else:
                visual_meshes.append(tm)

        import trimesh.util as tutil
        visual = tutil.concatenate(visual_meshes) if visual_meshes else None

        if collision_meshes:
            collision_hull = tutil.concatenate(collision_meshes).convex_hull
        elif visual_meshes:
            collision_hull = tutil.concatenate(visual_meshes).convex_hull
        else:
            collision_hull = None

        n_vis = sum(len(m.vertices) for m in visual_meshes)
        n_col = len(collision_hull.vertices) if collision_hull else 0
        logger.info(
            f"    → {len(visual_meshes)} vis submesh(es) {n_vis}v,"
            f" collision hull {n_col}v"
        )
        return ExtractionResult(
            category=category, model=model,
            visual=visual, collision_hull=collision_hull,
        )


# ── result container ─────────────────────────────────────────────────────────

class ExtractionResult:
    """Holds extracted meshes for one B1K asset."""

    def __init__(
        self,
        category: str,
        model: str,
        visual,          # trimesh.Trimesh | None
        collision_hull,  # trimesh.Trimesh | None  (convex hull)
    ):
        self.category = category
        self.model = model
        self.visual = visual
        self.collision_hull = collision_hull

    # ------------------------------------------------------------------
    def save(
        self,
        output_dir: Path,
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """Write meshes to *output_dir*.

        Args:
            output_dir: directory for this asset (created if absent).
            formats: list of formats to export, e.g. ``["obj", "glb", "stl"]``.
                     Defaults to ``["obj"]``.

        Returns:
            dict mapping role ("visual", "collision") × format → Path.
        """
        if formats is None:
            formats = ["obj"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        for fmt in formats:
            fmt = fmt.lower().lstrip(".")

            if self.visual is not None:
                p = output_dir / f"visual.{fmt}"
                self.visual.export(str(p))
                written[f"visual.{fmt}"] = p

            if self.collision_hull is not None:
                p = output_dir / f"collision.{fmt}"
                self.collision_hull.export(str(p))
                written[f"collision.{fmt}"] = p

        return written

    # ------------------------------------------------------------------
    @property
    def visual_bounds(self) -> np.ndarray | None:
        """2×3 array [[xmin,ymin,zmin],[xmax,ymax,zmax]] or None."""
        return self.visual.bounds if self.visual is not None else None

    @property
    def visual_extents(self) -> np.ndarray | None:
        """[dx, dy, dz] extent in metres, or None."""
        b = self.visual_bounds
        return (b[1] - b[0]) if b is not None else None

    def __repr__(self) -> str:  # pragma: no cover
        nv = len(self.visual.vertices) if self.visual else 0
        nc = len(self.collision_hull.vertices) if self.collision_hull else 0
        return (
            f"ExtractionResult({self.category}/{self.model},"
            f" visual={nv}v, collision_hull={nc}v)"
        )
