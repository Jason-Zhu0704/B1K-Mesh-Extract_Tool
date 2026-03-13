# b1k-mesh-extractor

Convert [BEHAVIOR-1K](https://behavior.stanford.edu/) encrypted USD assets into standard mesh formats (OBJ, GLB, STL, PLY) suitable for use in **Drake**, MuJoCo, Blender, or any other robotics/3D tool.

## Why this exists

BEHAVIOR-1K ships 8,600+ high-quality 3D object assets in encrypted `.usd` files locked to the OmniGibson/Isaac Sim runtime. Other simulators (Drake, MuJoCo, Gazebo …) cannot open these files. This tool uses OmniGibson as a decryption shim: it loads each asset, reads the geometry from memory via the OpenUSD Python API, and exports clean mesh files in any format trimesh supports.

**What you get per asset:**
| File | Content |
|------|---------|
| `visual.obj` | Full visual mesh in body-local coordinates |
| `collision.obj` | Convex hull of the USD collision geometry (Drake-ready) |
| `index.json` | Asset registry with bbox, paths, and metadata |

**Accuracy:** mesh bounds match B1K's own `metadata.json` to < 0.0001 m (verified).

## Requirements

| Dependency | Notes |
|-----------|-------|
| Python ≥ 3.10 | |
| [BEHAVIOR-1K dataset](https://behavior.stanford.edu/knowledgebase/download) | Licensed; free for research |
| [OmniGibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html) + Isaac Sim | Required for USD decryption |
| `trimesh` | `pip install trimesh` |
| `numpy` | |
| `drake` *(optional)* | Only needed to use extracted meshes in Drake |

> **GPU required** — OmniGibson/Isaac Sim needs a CUDA-capable GPU.

## Installation

```bash
git clone https://github.com/your-org/b1k-mesh-extractor.git
cd b1k-mesh-extractor
pip install -e .
```

## Quick start

```bash
# Extract one asset → OBJ (default)
b1k-extract --category bed --model fnzxxr --output ./b1k_meshes

# Extract entire category, multiple formats
b1k-extract --category bed --formats obj glb stl --output ./b1k_meshes

# Extract everything (≈8,600 assets; ~30 s/asset → several days)
b1k-extract --output ./b1k_meshes

# List categories
b1k-extract --list-categories
```

Output layout:
```
b1k_meshes/
  index.json                    ← asset registry
  objects/
    bed/
      fnzxxr/
        visual.obj              ← full visual mesh
        collision.obj           ← convex-hull collision mesh
```

## Python API

```python
import omnigibson as og
from omnigibson.macros import gm
from b1k_extractor.extractor import AssetExtractor
from pathlib import Path

# Init OmniGibson (headless)
gm.HEADLESS = True
gm.ENABLE_OBJECT_STATES = True
env = og.Environment(configs={"scene": {"type": "Scene"}, "objects": []})

extractor = AssetExtractor(env)
result = extractor.extract("bed", "fnzxxr")

print(result)                  # ExtractionResult(bed/fnzxxr, visual=556v, collision_hull=85v)
print(result.visual_extents)   # [2.235, 1.632, 0.673]  metres

result.save(Path("output/bed/fnzxxr"), formats=["obj", "glb"])
env.close()
```

## Using extracted meshes in Drake

```python
from pydrake.geometry import Convex, Mesh
from pydrake.math import RigidTransform

# Collision (convex hull — fast, exact for contact)
col = Convex("b1k_meshes/objects/bed/fnzxxr/collision.obj", scale=1.0)
plant.RegisterCollisionGeometry(body, RigidTransform(), col, "bed_col", props)

# Visual (full mesh)
vis = Mesh("b1k_meshes/objects/bed/fnzxxr/visual.obj", scale=1.0)
plant.RegisterVisualGeometry(body, RigidTransform(), vis, "bed_vis", rgba)
```

## How it works

1. OmniGibson loads the encrypted `.usd` file (requires a valid B1K license key at `datasets/omnigibson.key`).
2. The geometry is read from the live USD stage via `pxr.UsdGeom.Mesh`.
3. Each mesh prim is transformed from its USD-local frame into **body-local space** (relative to the object root prim) using `UsdGeom.XformCache`.
4. Visual submeshes are concatenated and exported directly.
5. Collision submeshes are merged and convex-hull'd via `trimesh` for Drake compatibility.
6. An `index.json` registry records bbox sizes and file paths for all extracted assets.

## Coordinate convention

All exported meshes are in **body-local space**: the object root prim is the origin. In Drake terms, register the collision geometry with `RigidTransform()` (identity) on the free body; Drake will move the mesh with the body automatically.

## License

MIT — code only. The BEHAVIOR-1K assets are licensed separately by Stanford.
See [BEHAVIOR-1K Terms of Use](https://behavior.stanford.edu/knowledgebase/download).
