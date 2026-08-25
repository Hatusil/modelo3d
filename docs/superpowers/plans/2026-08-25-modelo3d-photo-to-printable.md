# modelo3d Photo-to-Printable Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Google Colab notebook (`modelo3d.ipynb`) that launches a Spanish-language Gradio app converting one photo (or three aligned views) into a watertight, slicer-ready binary STL.

**Architecture:** One notebook with tagged code cells: intro markdown → setup (GPU gate, deps, pinned weight download) → Gradio app. Pure CPU logic (validation, mesh repair, scaling, base union, export) lives in cells tagged `core`; local pytest extracts and executes `core` cells from the notebook JSON so they are genuinely TDD-tested without a GPU. GPU-only code (Hunyuan3D engines, Gradio UI, self-test) is covered by structural tests locally plus an in-product self-test and a manual Colab T4 protocol before handoff.

**Tech Stack:** Python 3, nbformat, Gradio, trimesh + PyMeshFix + manifold3d (via trimesh.boolean), Pillow, NumPy, Hunyuan3D-2mini-Turbo (single photo) / Hunyuan3D-2mv (multiview) via Hugging Face, pytest.

## Global Constraints

- All end-user-facing text (notebook prose, UI copy, error messages, README) in neutral/professional Spanish.
- Output: binary STL (plus GLB for preview). Textures/color generation excluded.
- Exported mesh MUST be watertight; `verify_stl` enforces it (reload → `is_watertight` is true, positive volume, height matches target ±0.5 mm).
- Size presets: `"10cm"` → 100 mm (default), `"15cm"` → 150 mm, `"custom"` → user millimeters.
- Flat base optional, default ON.
- Engines: single-photo = `tencent/Hunyuan3D-2mini` subfolder `hunyuan3d-dit-v2-mini-turbo`; multiview = `tencent/Hunyuan3D-2mv` subfolder `hunyuan3d-dit-v2-mv`. Both pinned to exact Hugging Face revisions (Task 4).
- Multiview mode needs ≥2 usable views; below that, fall back to the single-photo engine and tell the user explicitly.
- CUDA OOM: exactly one automatic retry with low-VRAM parameters, then friendly failure.
- End user never sees a raw stacktrace; all failures map to `friendly_error()`.
- The notebook remains runnable as a single file (Colab fetches only the `.ipynb` from GitHub — no local imports inside it).
- License: MIT ("Copyright (c) 2026 modelo3d contributors").
- Dev-only additions beyond the approved layout: `tests/` directory (test harness). The product itself stays exactly README.md + modelo3d.ipynb + LICENSE.
- Canonical notebook-editing method everywhere: write cell source to `/tmp/opencode/cell_<name>.py`, then merge with the nbformat heredoc shown in Task 1 Step 5. Never hand-edit JSON.

---

## File Map

| File | Responsibility |
|---|---|
| `modelo3d.ipynb` | The whole product: intro md, `setup` cell, `app` cell, `core` cells, `selftest` cell |
| `tests/conftest.py` | Loads notebook, executes `core` cells into a namespace, exposes `core` fixture |
| `tests/test_*.py` | Per-task pytest modules (pure-logic + structural assertions) |
| `README.md` | ES usage guide + Open-in-Colab badge |
| `LICENSE` | MIT |

Cell inventory (final state of `modelo3d.ipynb`):

| # | Type | Tags | Content |
|---|---|---|---|
| 0 | markdown | — | Intro (ES) |
| 1 | code | `setup` | GPU gate, installs, pinned downloads |
| 2 | code | `core` | Config + `resolve_size_preset` |
| 3 | code | `core` | `validate_image`, `mask_fraction`, `ERRORS_ES`, `friendly_error` |
| 4 | code | `core` | `repair_mesh`, `normalize_mesh`, `add_flat_base`, `export_stl`, `verify_stl` |
| 5 | code | `core` | `choose_engine`, `select_mv_strategy`, `GEN_PARAMS`, `GEN_PARAMS_FAST`, `generate_with_retry` |
| 6 | code | `app` | `build_app()` Gradio UI + handler |
| 7 | code | `selftest` | `make_sample_image` + end-to-end self-test |

---

### Task 1: Notebook skeleton, test harness, size presets

**Files:**
- Create: `modelo3d.ipynb`
- Create: `tests/conftest.py`
- Test: `tests/test_presets.py`

**Interfaces:**
- Consumes: nothing.
- Produces: notebook file with intro markdown cell; `conftest.core` fixture (dict namespace of all `core` cells); `resolve_size_preset(preset: str, custom_mm: int | None) -> int`.

- [ ] **Step 1: Create dev environment**

```bash
cd ~/projects/modelo3d && python3 -m venv .venv && . .venv/bin/activate && pip install pytest nbformat trimesh pymeshfix manifold3d pillow numpy ruff
```

Expected: clean install. Add `.venv/` to `.gitignore` (create file with `.venv/\n__pycache__/\n*.stl\n*.glb\n`).

- [ ] **Step 2: Write the failing test**

Create `tests/conftest.py`:

```python
import nbformat
import pytest

CORE_TAG = "core"
NOTEBOOK = "modelo3d.ipynb"


def load_core_namespace():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ns: dict = {}
    for cell in nb.cells:
        tags = cell.get("metadata", {}).get("tags", [])
        if cell.cell_type == "code" and CORE_TAG in tags:
            exec(compile(cell.source, f"<cell:{cell.id}>", "exec"), ns)
    return ns


@pytest.fixture(scope="session")
def core():
    return load_core_namespace()
```

Create `tests/test_presets.py`:

```python
import pytest


def test_default_preset_is_100mm(core):
    assert core["resolve_size_preset"]("10cm", None) == 100


def test_15cm_preset(core):
    assert core["resolve_size_preset"]("15cm", None) == 150


def test_custom_preset(core):
    assert core["resolve_size_preset"]("custom", 73) == 73


def test_custom_requires_value(core):
    with pytest.raises(ValueError, match="milímetros"):
        core["resolve_size_preset"]("custom", None)


def test_unknown_preset(core):
    with pytest.raises(ValueError):
        core["resolve_size_preset"]("20cm", None)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `. .venv/bin/activate && python -m pytest tests/test_presets.py -v`
Expected: FAIL / ERROR — `FileNotFoundError: modelo3d.ipynb`.

- [ ] **Step 4: Create the notebook skeleton with intro cell**

Write intro source to `/tmp/opencode/intro.md`, then run:

```bash
python3 - <<'EOF'
import nbformat as nbf

intro = open("/tmp/opencode/intro.md").read()
nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell(intro))
nbf.write(nb, "modelo3d.ipynb")
print("cells:", len(nb.cells))
EOF
```

Intro markdown content (`/tmp/opencode/intro.md`) — complete text:

```markdown
# 🗿 modelo3d — De foto a modelo 3D imprimible

Convertí una foto (o tres vistas del mismo objeto) en un archivo **STL listo para imprimir**, sin saber nada de programación.

## Qué necesitás
- Una cuenta de Google (gratis).
- Una foto del objeto: buena luz, fondo liso, un solo objeto centrado.

## Cuánto tarda
- **Primera vez:** 5–8 minutos de instalación automática (solo una vez por sesión).
- **Cada modelo:** entre 30 segundos y 2 minutos.

## Antes de empezar
1. Hacé clic en **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU → Guardar**.
2. Ejecutá la celda de instalación de abajo y esperá el mensaje ✅.
3. La aplicación va a aparecer al final de la página.

⚠️ **Importante:** cuando la sesión de Colab se cierre, los archivos se borran. Descargá tu STL apenas lo generes.
```

- [ ] **Step 5: Add core cell #1 (config + presets)**

Write to `/tmp/opencode/cell_config.py`:

```python
# --- Configuración general ---
SIZE_PRESETS_MM = {"10cm": 100, "15cm": 150}
DEFAULT_PRESET = "10cm"


def resolve_size_preset(preset: str, custom_mm: int | None) -> int:
    """Devuelve la altura objetivo en milímetros."""
    if preset in SIZE_PRESETS_MM:
        return SIZE_PRESETS_MM[preset]
    if preset == "custom":
        if not custom_mm or custom_mm <= 0:
            raise ValueError("Ingresá un alto en milímetros válido (mayor a 0).")
        return int(custom_mm)
    raise ValueError(f"Tamaño desconocido: {preset}")
```

Merge it (this exact snippet pattern is reused in every later task — change only the source path):

```bash
python3 - <<'EOF'
import nbformat as nbf

nb = nbf.read("modelo3d.ipynb", as_version=4)
src = open("/tmp/opencode/cell_config.py").read()
cell = nbf.v4.new_code_cell(src)
cell["metadata"]["tags"] = ["core"]
nb.cells.append(cell)
nbf.write(nb, "modelo3d.ipynb")
print("cells:", len(nb.cells))
EOF
```

Expected: `cells: 2`.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_presets.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add .gitignore modelo3d.ipynb tests/conftest.py tests/test_presets.py
git commit -m "feat: notebook skeleton with core-cell test harness and size presets"
```

---

### Task 2: Photo validation and user-facing error catalog

**Files:**
- Modify: `modelo3d.ipynb` (append core cell)
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `validate_image(img: PIL.Image.Image) -> None` — raises `ValueError` whose message is the Spanish text from `ERRORS_ES`.
  - `mask_fraction(mask: np.ndarray) -> float` — foreground fraction of a boolean mask.
  - `check_mask_sane(frac: float) -> None` — raises `ValueError` if the segmented object looks wrong.
  - `ERRORS_ES: dict[str, str]` — canonical Spanish messages keyed by code.
  - `friendly_error(exc: Exception) -> str` — maps any exception to a safe Spanish message (never leaks internals).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validation.py`:

```python
import numpy as np
from PIL import Image
import pytest


def make_img(w=512, h=512, color=(30, 30, 30)):
    return Image.new("RGB", (w, h), color)


def test_valid_image_passes(core):
    core["validate_image"](make_img())


def test_too_small_rejected(core):
    with pytest.raises(ValueError, match="chica"):
        core["validate_image"](make_img(200, 200))


def test_grayscale_converted_not_rejected(core):
    img = Image.new("L", (600, 400))
    core["validate_image"](img)


def test_mask_fraction(core):
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    assert core["mask_fraction"](mask) == pytest.approx(0.04)


def test_mask_all_background(core):
    with pytest.raises(ValueError, match="objeto"):
        core["check_mask_sane"](0.001)


def test_mask_all_foreground(core):
    with pytest.raises(ValueError, match="fondo"):
        core["check_mask_sane"](0.97)


def test_friendly_error_hides_internals(core):
    err = core["friendly_error"](RuntimeError("CUDA malloc 0xdeadbeef secret"))
    assert "0xdeadbeef" not in err
    assert err  # non-empty Spanish message
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_validation.py -v`
Expected: FAIL — `KeyError: 'validate_image'`.

- [ ] **Step 3: Implement the core cell**

Write to `/tmp/opencode/cell_validation.py`:

```python
# --- Validación de fotos y mensajes ---
import numpy as np
from PIL import Image

MIN_SIDE_PX = 256

ERRORS_ES = {
    "too_small": "La foto es muy chica. Usá una imagen de al menos 256 píxeles por lado.",
    "unreadable": "No pudimos leer la imagen. Probá con otro archivo JPG o PNG.",
    "no_object": "No detectamos ningún objeto en la foto. Revisá que el objeto se vea completo y con buen contraste contra el fondo.",
    "bad_cutout": "El recorte del objeto quedó raro. Sacá la foto con el objeto centrado sobre un fondo liso, sin manos y sin que se corte con el borde.",
}

FALLBACK_ERROR_ES = (
    "Algo salió mal generando el modelo. Probá de nuevo; si sigue fallando, "
    "probá con otra foto."
)


def _to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def validate_image(img: Image.Image) -> None:
    try:
        img = _to_rgb(img)
        w, h = img.size
    except Exception as exc:
        raise ValueError(ERRORS_ES["unreadable"]) from exc
    if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
        raise ValueError(ERRORS_ES["too_small"])


def mask_fraction(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / float(mask.size)


def check_mask_sane(fraction: float) -> None:
    if fraction < 0.01:
        raise ValueError(ERRORS_ES["no_object"])
    if fraction > 0.90:
        raise ValueError(ERRORS_ES["bad_cutout"])


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, ValueError) and str(exc) in ERRORS_ES.values():
        return str(exc)
    return FALLBACK_ERROR_ES
```

Merge (Task 1 Step 5 snippet, source `/tmp/opencode/cell_validation.py`). Expected: `cells: 3`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_validation.py tests/test_presets.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add modelo3d.ipynb tests/test_validation.py
git commit -m "feat: photo validation with Spanish error catalog"
```

---

### Task 3: Geometry core — repair, normalize, base, export

**Files:**
- Modify: `modelo3d.ipynb` (append core cell)
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh` — watertight guaranteed or raises `ValueError`.
  - `normalize_mesh(mesh, target_height_mm: float) -> trimesh.Trimesh` — uniform-scaled so Z height equals target, resting on Z=0, centered on XY origin.
  - `add_flat_base(mesh, height_mm: float = 3.0) -> trimesh.Trimesh` — cylinder pedestal unioned underneath.
  - `export_stl(mesh, path: str) -> str` — writes binary STL, returns path.
  - `verify_stl(path: str, target_height_mm: float, tol_mm: float = 0.5) -> trimesh.Trimesh` — reloads and asserts watertight + positive volume + height within tolerance; returns the reloaded mesh.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_geometry.py`:

```python
import numpy as np
import trimesh
import pytest


def broken_sphere() -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=3, radius=25.0)
    m.faces = np.delete(m.faces, slice(0, len(m.faces) // 5), axis=0)  # punch holes
    return m


def test_repair_makes_watertight(core):
    fixed = core["repair_mesh"](broken_sphere())
    assert fixed.is_watertight
    assert fixed.volume > 0


def test_normalize_height_and_grounding(core):
    m = trimesh.creation.icosphere(radius=10.0)
    out = core["normalize_mesh"](m, 120.0)
    extents = out.extents
    assert extents[2] == pytest.approx(120.0, abs=0.5)
    assert out.bounds[0][2] == pytest.approx(0.0, abs=0.1)
    assert out.bounds[:, :2].mean(axis=0) == pytest.approx([0, 0], abs=0.1)


def test_base_adds_volume_and_keeps_watertight(core):
    m = core["normalize_mesh"](trimesh.creation.icosphere(radius=10.0), 100.0)
    based = core["add_flat_base"](m, height_mm=3.0)
    assert based.is_watertight
    assert based.volume > m.volume


def test_export_and_verify_roundtrip(core, tmp_path):
    m = core["normalize_mesh"](trimesh.creation.icosphere(radius=10.0), 100.0)
    p = str(tmp_path / "out.stl")
    assert core["export_stl"](m, p) == p
    reloaded = core["verify_stl"](p, 100.0)
    assert reloaded.is_watertight


def test_verify_rejects_wrong_scale(core, tmp_path):
    m = core["normalize_mesh"](trimesh.creation.icosphere(radius=10.0), 100.0)
    p = str(tmp_path / "wrong.stl")
    core["export_stl"](m, p)
    with pytest.raises(ValueError, match="alto"):
        core["verify_stl"](p, 150.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_geometry.py -v`
Expected: FAIL — `KeyError: 'repair_mesh'`.

- [ ] **Step 3: Implement the core cell**

Write to `/tmp/opencode/cell_geometry.py`:

```python
# --- Núcleo geométrico: reparar, escalar, base, exportar ---
import numpy as np
import trimesh
import trimesh.boolean

BASE_HEIGHT_MM = 3.0
BASE_MARGIN = 0.95  # radio del pedestal relativo al alcance XY del modelo


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        try:
            import pymeshfix

            fixed = pymeshfix.clean_from_arrays(mesh.vertices, mesh.faces)
            mesh = trimesh.Trimesh(fixed[0], fixed[1], process=False)
        except Exception as exc:
            raise ValueError(
                "El modelo salió con agujeros que no pudimos reparar. "
                "Probá generar de nuevo con otra foto."
            ) from exc
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def normalize_mesh(
    mesh: trimesh.Trimesh, target_height_mm: float
) -> trimesh.Trimesh:
    m = mesh.copy()
    height = float(m.extents[2])
    if height <= 0:
        raise ValueError("La geometría generada es plana e inválida.")
    m.apply_scale(target_height_mm / height)
    m.apply_translation(-m.bounds[0])          # apoyar en Z=0
    center_xy = m.bounds.mean(axis=0)[:2]
    m.apply_translation([-center_xy[0], -center_xy[1], 0])
    return m


def _pedestal(radius_mm: float, height_mm: float) -> trimesh.Trimesh:
    cyl = trimesh.creation.cylinder(
        radius=radius_mm, height=height_mm, sections=64
    )
    cyl.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    )
    cyl.apply_translation([0, 0, height_mm / 2.0])
    return cyl


def add_flat_base(
    mesh: trimesh.Trimesh, height_mm: float = BASE_HEIGHT_MM
) -> trimesh.Trimesh:
    xy_span = float(max(mesh.extents[0], mesh.extents[1]))
    pedestal = _pedestal(xy_span * BASE_MARGIN * 0.5, height_mm)
    merged = trimesh.boolean.union([mesh, pedestal], engine="manifold")
    return merged


def export_stl(mesh: trimesh.Trimesh, path: str) -> str:
    mesh.export(path, file_type="stl")
    return path


def verify_stl(
    path: str, target_height_mm: float, tol_mm: float = 0.5
) -> trimesh.Trimesh:
    reloaded = trimesh.load(path, force="mesh")
    if not reloaded.is_watertight:
        raise ValueError("El STL exportado tiene agujeros.")
    if reloaded.volume <= 0:
        raise ValueError("El STL exportado está vacío.")
    height = float(reloaded.extents[2])
    if abs(height - target_height_mm) > tol_mm:
        raise ValueError(
            f"El STL mide {height:.1f} mm de alto en vez de {target_height_mm:.1f} mm."
        )
    return reloaded
```

Merge (source `/tmp/opencode/cell_geometry.py`). Expected: `cells: 4`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add modelo3d.ipynb tests/test_geometry.py
git commit -m "feat: watertight mesh repair, normalization, base and STL export"
```

---

### Task 4: Setup cell — GPU gate, dependencies, pinned downloads

**Files:**
- Modify: `modelo3d.ipynb` (insert setup cell after intro)
- Test: `tests/test_structure_setup.py`

**Interfaces:**
- Consumes: nothing.
- Produces (names defined at notebook runtime for later cells): `DEVICE_OK: bool`, `MODEL_SINGLE: tuple[str, str]`, `MODEL_MULTI: tuple[str, str]`, `HF_REVISIONS: dict[str, str]` (40-hex shas), `REPO_DIR: str`, `ensure_engines()` (lazy loader returning `(mini_pipeline, mv_pipeline_or_None)`).

- [ ] **Step 1: Write the failing structural test**

Create `tests/test_structure_setup.py`:

```python
import re
import nbformat

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _cell_by_tag(tag):
    nb = nbformat.read("modelo3d.ipynb", as_version=4)
    for cell in nb.cells:
        if tag in cell.get("metadata", {}).get("tags", []):
            return cell
    raise AssertionError(f"missing cell tagged {tag}")


def test_setup_cell_exists():
    src = _cell_by_tag("setup").source
    assert "tencent/Hunyuan3D-2mini" in src
    assert "tencent/Hunyuan3D-2mv" in src
    assert "hunyuan3d-dit-v2-mini-turbo" in src
    assert "Entorno de ejecución" in src  # GPU-gate message in Spanish


def test_revision_pins_are_real_shas():
    src = _cell_by_tag("setup").source
    shas = re.findall(r'"revision":\s*"([^"]+)"', src)
    assert len(shas) == 2
    for sha in shas:
        assert SHA_RE.match(sha), f"not a pinned sha: {sha}"
```

- [ ] **Step 2: Fetch the real revision SHAs (no fabricated hashes)**

```bash
curl -s https://huggingface.co/api/models/tencent/Hunyuan3D-2mini | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])"
curl -s https://huggingface.co/api/models/tencent/Hunyuan3D-2mv | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])"
```

Expected: two distinct 40-char hex strings. Record them as `<SHA_MINI>` and `<SHA_MV>` for Step 3.

- [ ] **Step 3: Implement the setup cell**

Write to `/tmp/opencode/cell_setup.py`, substituting the two real SHAs from Step 2:

```python
# --- Instalación (ejecutar primero) ---
import sys

if not __import__("torch").cuda.is_available():
    raise RuntimeError(
        "GPU no activada. Andá a 'Entorno de ejecución → Cambiar tipo de entorno "
        "de ejecución → T4 GPU', guardá, y volvé a ejecutar esta celda."
    )

print("✅ GPU detectada:", __import__("torch").cuda.get_device_name(0))

REPO_URL = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git"
REPO_DIR = "/content/Hunyuan3D-2"
MODEL_SINGLE = ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo")
MODEL_MULTI = ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv")
HF_REVISIONS = {
    "mini": "<SHA_MINI>",
    "mv": "<SHA_MV>",
}

import subprocess  # noqa: E402

subprocess.run(
    ["bash", "-lc",
     f'test -d {REPO_DIR} || git clone --depth 1 {REPO_URL} {REPO_DIR}'],
    check=True,
)
subprocess.run(
    ["bash", "-lc",
     f'pip install -q -r {REPO_DIR}/requirements.txt '
     'pyrembg trimesh pymeshfix manifold3d gradio'],
    check=True,
)
sys.path.insert(0, REPO_DIR)

from huggingface_hub import snapshot_download  # noqa: E402

print("⬇️ Descargando modelo de una foto (~1 GB)...")
snapshot_download(
    repo_id=MODEL_SINGLE[0],
    revision=HF_REVISIONS["mini"],
    allow_patterns=[f"{MODEL_SINGLE[1]}/*"],
)
print("⬇️ Descargando modelo multivista (~2 GB)...")
snapshot_download(
    repo_id=MODEL_MULTI[0],
    revision=HF_REVISIONS["mv"],
    allow_patterns=[f"{MODEL_MULTI[1]}/*"],
)

_ENGINES = None


def ensure_engines():
    """Carga diferida de los pipelines (solo forma, sin texturas)."""
    global _ENGINES
    if _ENGINES is None:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        mini = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_SINGLE[0], subfolder=MODEL_SINGLE[1]
        )
        mini.to("cuda")
        try:
            mv = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                MODEL_MULTI[0], subfolder=MODEL_MULTI[1]
            )
            mv.to("cuda")
        except Exception:
            print("⚠️ Motor multivista no disponible; se usará el de una foto.")
            mv = None
        _ENGINES = (mini, mv)
    return _ENGINES


print("✅ Instalación lista. Ejecutá la celda de abajo para abrir la app.")
```

Insert it as cell index 1 (after intro) using nbformat (`nb.cells.insert(1, cell)`), tag `setup`. Expected: `cells: 5`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_structure_setup.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add modelo3d.ipynb tests/test_structure_setup.py
git commit -m "feat: setup cell with GPU gate, pinned HF revisions and lazy engines"
```

---

### Task 5: Engine layer — selection, multiview strategy, OOM retry

**Files:**
- Modify: `modelo3d.ipynb` (append core cell)
- Test: `tests/test_engine_logic.py`

**Interfaces:**
- Consumes: `ensure_engines()` (runtime), `ValueError` messages from Task 2.
- Produces:
  - `GEN_PARAMS: dict` (default: `steps=50, octree=256`), `GEN_PARAMS_FAST: dict` (`steps=30, octree=192`).
  - `choose_engine(n_views: int) -> str` — `"mv"` if ≥2 else `"single"`.
  - `select_mv_strategy(mv_available: bool, n_views: int) -> tuple[str, str]` — returns `(engine, notice_es)`; falls back to `"single"` with an explicit Spanish notice when mv unavailable.
  - `generate_with_retry(run_fn, params) -> object` — calls `run_fn(params)`; on OOM retries ONCE with `GEN_PARAMS_FAST`; re-raises anything else.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_engine_logic.py`:

```python
import pytest


class FakeOOM(Exception):
    pass


def test_choose_engine_threshold(core):
    assert core["choose_engine"](1) == "single"
    assert core["choose_engine"](2) == "mv"
    assert core["choose_engine"](3) == "mv"
    assert core["choose_engine"](0) == "single"


def test_mv_fallback_is_explicit(core):
    engine, notice = core["select_mv_strategy"](False, 3)
    assert engine == "single"
    assert notice  # non-empty Spanish explanation


def test_mv_used_when_available(core):
    engine, notice = core["select_mv_strategy"](True, 3)
    assert engine == "mv"
    assert notice == ""


def test_retry_after_oom_uses_fast_params(core):
    calls = []

    def flaky(params):
        calls.append(params)
        if len(calls) == 1:
            raise FakeOOM("CUDA out of memory")
        return "mesh"

    out = core["generate_with_retry"](flaky, core["GEN_PARAMS"], oom_exc=FakeOOM)
    assert out == "mesh"
    assert calls[0] == core["GEN_PARAMS"]
    assert calls[1] == core["GEN_PARAMS_FAST"]


def test_no_retry_on_other_errors(core):
    def boom(params):
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        core["generate_with_retry"](boom, core["GEN_PARAMS"], oom_exc=FakeOOM)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_engine_logic.py -v`
Expected: FAIL — `KeyError: 'choose_engine'`.

- [ ] **Step 3: Implement the core cell**

Write to `/tmp/opencode/cell_engine.py`:

```python
# --- Selección de motor y reintento por memoria ---
GEN_PARAMS = {"num_inference_steps": 50, "octree_resolution": 256}
GEN_PARAMS_FAST = {"num_inference_steps": 30, "octree_resolution": 192}


def choose_engine(n_views: int) -> str:
    return "mv" if n_views >= 2 else "single"


def select_mv_strategy(mv_available: bool, n_views: int) -> tuple[str, str]:
    wanted = choose_engine(n_views)
    if wanted == "mv" and not mv_available:
        return "single", (
            "El modo varias fotos no está disponible en esta sesión; "
            "usamos el motor de una sola foto."
        )
    return wanted, ""


class OutOfMemoryError_(Exception):
    """Marcador local; en Colab se usa torch.cuda.OutOfMemoryError."""


def generate_with_retry(run_fn, params: dict, oom_exc: type = Exception):
    import torch  # solo dentro de funciones GPU

    oom = getattr(torch.cuda, "OutOfMemoryError", oom_exc)
    try:
        return run_fn(params)
    except oom:
        print("⚠️ Sin memoria suficiente; reintentando en calidad reducida...")
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        return run_fn({**params, **GEN_PARAMS_FAST})
```

Note: `generate_with_retry` imports torch lazily so the pure-logic tests inject `oom_exc=FakeOOM` — wait, the signature above ignores the injected exception when torch IS importable. Fix: prefer the injected `oom_exc` when the caller passes a non-default value. Replace the body between the signature and `try:` with:

```python
    try:
        import torch
        oom = oom_exc if oom_exc is not Exception else getattr(
            torch.cuda, "OutOfMemoryError", RuntimeError
        )
    except ImportError:
        oom = oom_exc
```

…and delete the earlier `oom = ...` line accordingly. (Final cell keeps this corrected version.)

Merge (source `/tmp/opencode/cell_engine.py`). Expected: `cells: 6`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: 25 passed.

- [ ] **Step 5: Commit**

```bash
git add modelo3d.ipynb tests/test_engine_logic.py
git commit -m "feat: engine selection, explicit mv fallback and single OOM retry"
```

---

### Task 6: Gradio app cell

**Files:**
- Modify: `modelo3d.ipynb` (append app cell)
- Test: `tests/test_structure_app.py` + extend `tests/test_engine_logic.py`

**Interfaces:**
- Consumes: everything above (`validate_image`, `mask_fraction`, `check_mask_sane`, `repair_mesh`, `normalize_mesh`, `add_flat_base`, `export_stl`, `verify_stl`, `resolve_size_preset`, `select_mv_strategy`, `generate_with_retry`, `ensure_engines`, `friendly_error`).
- Produces: `build_app() -> gradio.Blocks`; `run_pipeline(single, front, left, back, modo, preset, custom_mm, want_base, progress) -> tuple[glb_path, stl_path, status_str]`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_structure_app.py` (new file):

```python
import nbformat


def _src(tag):
    nb = nbformat.read("modelo3d.ipynb", as_version=4)
    for cell in nb.cells:
        if tag in cell.get("metadata", {}).get("tags", []):
            return cell.source
    raise AssertionError(f"missing {tag}")


def test_app_labels_are_spanish():
    src = _src("app")
    for label in [
        '"Una foto"', '"Varias fotos"', '"Generar"', '"Descargar STL"',
        '"Consejos para la foto"', '"Tamaño impreso"',
    ]:
        assert label in src, label


def test_app_never_shows_raw_traceback():
    src = _src("app")
    assert "friendly_error" in src
    assert "raise" not in src.split("def run_pipeline")[1].split("except ValueError")[0]
```

And add to `tests/test_engine_logic.py`:

```python
def test_run_pipeline_glue_exists(core):
    assert callable(core["run_pipeline"])
```

(Note: `run_pipeline` is defined in the app cell but must ALSO carry the `core` tag so the fixture sees it — see Step 3: the cell gets tags `["app", "core"]`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_structure_app.py -v`
Expected: FAIL — missing `app` tag.

- [ ] **Step 3: Implement the app cell**

Write to `/tmp/opencode/cell_app.py`:

```python
# --- Aplicación web ---
import tempfile
import gradio as gr

PHOTO_TIPS_ES = (
    "- Buena luz, sin flash directo\n"
    "- Fondo liso y de color parejo\n"
    "- Un solo objeto, centrado y completo\n"
    "- Sin manos sosteniendo el objeto\n"
    "- En modo varias fotos: misma distancia y altura en las tres tomas"
)


def _segment(img):
    """Quita el fondo y devuelve (imagen RGB, máscara bool)."""
    from hy3dgen.shapegen.rembg import BackgroundRemover

    remover = BackgroundRemover()
    out = remover(img)
    import numpy as np

    rgba = np.array(out.convert("RGBA"))
    mask = rgba[..., 3] > 127
    return out.convert("RGB"), mask


def _call_engine(engine, images, progress):
    mini, mv = ensure_engines()

    def run(params):
        progress((0.4, "Generando geometría…"))
        if engine == "mv":
            mesh = mv(image=images, **params)[0]
        else:
            mesh = mini(image=images[0], **params)[0]
        return mesh.to_data() if hasattr(mesh, "to_data") else mesh

    import torch

    raw = generate_with_retry(run, GEN_PARAMS, oom_exc=torch.cuda.OutOfMemoryError)
    import trimesh

    return raw if isinstance(raw, trimesh.Trimesh) else trimesh.Trimesh(
        raw.vertices.detach().cpu().numpy(),
        raw.faces.detach().cpu().numpy(),
        process=False,
    )


def run_pipeline(single, front, left, back, modo, preset, custom_mm,
                 want_base, progress=gr.Progress()):
    imgs = []
    if modo == "Varias fotos":
        imgs = [im for im in (front, left, back) if im is not None]
        if not imgs:
            imgs = [single]
    else:
        imgs = [single]

    progress((0.05, "Revisando la foto…"))
    validate_image(imgs[0])
    seg = [_segment(im) for im in imgs]
    for _, mask in seg:
        check_mask_sane(mask_fraction(mask))
    imgs = [im for im, _ in seg]

    mini, mv = ensure_engines()
    engine, notice = select_mv_strategy(mv is not None, len(imgs))

    mesh = _call_engine(engine, imgs, progress)
    progress((0.7, "Reparando la malla…"))
    mesh = repair_mesh(mesh)

    target_mm = resolve_size_preset(preset, custom_mm)
    mesh = normalize_mesh(mesh, target_mm)
    if want_base:
        mesh = add_flat_base(mesh)

    progress((0.9, "Exportando STL…"))
    workdir = tempfile.mkdtemp(prefix="modelo3d_")
    stl_path = export_stl(mesh, f"{workdir}/modelo.stl")
    verify_stl(stl_path, target_mm)
    glb_path = f"{workdir}/modelo.glb"
    mesh.export(glb_path, file_type="glb")

    status = "✅ Modelo listo para imprimir."
    if notice:
        status += f" ({notice})"
    return glb_path, stl_path, status


def build_app():
    with gr.Blocks(title="modelo3d") as demo:
        gr.Markdown("## 🗿 De foto a modelo 3D imprimible")
        with gr.Accordion("Consejos para la foto", open=False):
            gr.Markdown(PHOTO_TIPS_ES)
        modo = gr.Radio(["Una foto", "Varias fotos"],
                        value="Una foto", label="Modo")
        single = gr.Image(type="pil", label="Foto del objeto")
        with gr.Row(visible=False) as fila_multi:
            front = gr.Image(type="pil", label="Frente")
            left = gr.Image(type="pil", label="Perfil izquierdo")
            back = gr.Image(type="pil", label="Espalda")
        modo.change(lambda m: gr.update(visible=m == "Varias fotos"),
                    modo, fila_multi)
        tamano = gr.Dropdown(["10cm", "15cm", "custom"],
                             value="10cm", label="Tamaño impreso")
        custom_mm = gr.Number(label="Milímetros (si elegís custom)",
                              precision=0)
        base_chk = gr.Checkbox(value=True,
                               label="Agregar base plana (recomendado)")
        btn = gr.Button("Generar", variant="primary")
        preview = gr.Model3D(label="Vista previa")
        archivo = gr.File(label="Descargar STL")
        estado = gr.Markdown()

        def wrapped(*args):
            try:
                return run_pipeline(*args)
            except Exception as exc:  # nunca mostrar traceback crudo
                raise gr.Error(friendly_error(exc))

        btn.click(wrapped,
                  [single, front, left, back, modo, tamano, custom_mm,
                   base_chk],
                  [preview, archivo, estado])
    return demo


demo = build_app()
demo.queue().launch(share=True, debug=False)
```

Merge with tags `["app", "core"]` (modify the merge snippet's tag line to `cell["metadata"]["tags"] = ["app", "core"]`). Expected: `cells: 7`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: 27 passed. (Local run does NOT execute the cell — `launch()` at module bottom would start a server. Guard it: the last two lines must be wrapped as `if os.environ.get("MODELO3D_IN_COLAB") == "1":` before merging; update `/tmp/opencode/cell_app.py` accordingly so local exec of `core` cells is side-effect free.)

- [ ] **Step 5: Commit**

```bash
git add modelo3d.ipynb tests/
git commit -m "feat: Spanish Gradio app wired to full pipeline with safe errors"
```

---

### Task 7: Self-test cell with deterministic sample image

**Files:**
- Modify: `modelo3d.ipynb` (append selftest cell)
- Test: `tests/test_structure_selftest.py` + `tests/test_sample_image.py`

**Interfaces:**
- Consumes: `_segment`, `run_pipeline` internals, `verify_stl`.
- Produces: `make_sample_image() -> PIL.Image` (deterministic), `run_self_test() -> bool`.

- [ ] **Step 1: Write failing tests**

`tests/test_sample_image.py`:

```python
import numpy as np


def test_sample_image_deterministic_and_usable(core):
    a = np.array(core["make_sample_image"]())
    b = np.array(core["make_sample_image"]())
    assert (a == b).all()
    assert a.shape[:2] == (768, 768)
    assert a.std() > 10  # enough contrast for background removal
```

`tests/test_structure_selftest.py`:

```python
import nbformat


def test_selftest_cell_runs_full_chain():
    nb = nbformat.read("modelo3d.ipynb", as_version=4)
    src = next(c.source for c in nb.cells
               if "selftest" in c.get("metadata", {}).get("tags", []))
    assert "make_sample_image" in src
    assert "verify_stl" in src
    assert "SELF TEST" in src  # bilingual summary marker
```

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest tests/test_sample_image.py tests/test_structure_selftest.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the selftest cell**

Write to `/tmp/opencode/cell_selftest.py`:

```python
# --- Modo prueba (recomendado antes del primer uso) ---
from PIL import Image, ImageDraw


def make_sample_image() -> Image.Image:
    """Imagen sintética determinista: figura oscura sobre fondo claro."""
    img = Image.new("RGB", (768, 768), (235, 235, 230))
    d = ImageDraw.Draw(img)
    d.ellipse([284, 84, 484, 284], fill=(70, 55, 45))     # cabeza
    d.rounded_rectangle([264, 264, 504, 684], radius=80,
                        fill=(90, 75, 60))                 # cuerpo
    return img


def run_self_test() -> bool:
    print("🧪 SELF TEST — probando el pipeline completo…")
    img = make_sample_image()
    glb, stl, status = run_pipeline(
        img, None, None, None, "Una foto", "10cm", None, True
    )
    ok = bool(stl) and "✅" in status
    print("SELF TEST:", "PASS ✅" if ok else "FAIL ❌", "—", status)
    return ok


if os.environ.get("MODELO3D_IN_COLAB") == "1":
    run_self_test()
```

(`import os` at the top of the cell.) Merge with tags `["selftest", "core"]`. Expected: `cells: 8`.

- [ ] **Step 4: Verify tests pass**

Run: `python -m pytest tests/ -v`
Expected: 30 passed.

- [ ] **Step 5: Commit**

```bash
git add modelo3d.ipynb tests/
git commit -m "feat: deterministic self-test proving the full pipeline"
```

---

### Task 8: README, LICENSE, final validation and Colab protocol

**Files:**
- Create: `README.md`, `LICENSE`
- Modify: none

**Interfaces:**
- Produces: distribution-ready repo; manual verification record for handoff.

- [ ] **Step 1: Write LICENSE (MIT)**

Standard MIT text, `Copyright (c) 2026 modelo3d contributors`.

- [ ] **Step 2: Write README.md (Spanish)**

Full content: what it is; 3-step usage (enable T4 → run setup → use app); Open-in-Colab badge pointing at `https://colab.research.google.com/github/<OWNER>/<REPO>/blob/main/modelo3d.ipynb`; photo guide (same tips as in-app); "Para desarrolladores" section documenting `pip install pytest nbformat trimesh pymeshfix manifold3d pillow numpy ruff` and `python -m pytest tests/ -v`; manual Colab T4 protocol checklist (object, pet, bust; multiview happy path; single-usable-view fallback; session-death recovery).

After `git remote add origin <url>` exists, replace `<OWNER>/<REPO>` mechanically:

```bash
REMOTE=$(git remote get-url origin | sed -E 's#.*github.com[:/]##; s/\.git$//')
sed -i "s#<OWNER>/<REPO>#${REMOTE}#" README.md
```

- [ ] **Step 3: Final automated checks**

```bash
python - <<'EOF'
import nbformat
nb = nbformat.read("modelo3d.ipynb", as_version=4)
nbformat.validate(nb)
assert len(nb.cells) == 8, len(nb.cells)
for c in nb.cells:
    if c.cell_type == "code":
        assert "/home/" not in c.source and "/Users/" not in c.source
print("notebook OK:", len(nb.cells), "cells")
EOF
python -m pytest tests/ -v      # Expected: 30 passed
. .venv/bin/activate && ruff check tests/
```

- [ ] **Step 4: Manual Colab T4 protocol (blocking handoff gate)**

Execute in Colab: all five cases from README checklist. Record results (pass/fail + notes) in the PR description. Do NOT ship to the friend until all five pass.

- [ ] **Step 5: Commit and push**

```bash
git add README.md LICENSE
git commit -m "docs: Spanish usage guide, MIT license and Colab test protocol"
git push -u origin main
```

---

## Self-Review Record

1. **Spec coverage:** presets ✓ (T1), validation+messages ✓ (T2), repair/normalize/base/export ✓ (T3), GPU gate+pins ✓ (T4), mv≥2 fallback explicit ✓ (T5), OOM single retry ✓ (T5), foolproof UI+accordion+progress ✓ (T6), self-test ✓ (T7), README/badge/MIT/manual protocol ✓ (T8). GLB preview ✓ (T6). Sessions-expiry warning ✓ (intro, T1).
2. **Placeholder scan:** `<SHA_MINI>/<SHA_MV>` are filled by an executable Step (T4 S2) from the live HF API — not invented values. `<OWNER>/<REPO>` replaced by a mechanical sed driven by the actual git remote (T8 S2). No other TBD/TODO patterns.
3. **Type consistency:** `resolve_size_preset(str, int|None)->int` used in T6 as declared in T1; `generate_with_retry(run_fn, params, oom_exc)` signature consistent between T5 definition and T6 usage (positional `run, GEN_PARAMS, oom_exc=`); `select_mv_strategy` returns `(str, str)` in both T5 and T6; `verify_stl(path, target)` called identically in T3 tests and T6/T7.
