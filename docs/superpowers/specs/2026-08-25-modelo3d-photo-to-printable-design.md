# modelo3d — Photo-to-Printable 3D Pipeline: Design

**Status:** Approved (2026-08-25)
**Audience of this doc:** implementation planner and implementer.

## Goal

A Google Colab notebook that launches a simple web app converting photos into
watertight, slicer-ready STL files. The end user is a non-technical person who
only clicks buttons in Spanish; the maintainer (repo owner) can adjust
parameters in code.

## Approved Decisions

| Topic | Decision |
|---|---|
| Primary use | Varied objects: people/busts, pets, figurines, objects |
| End-user interface | Gradio web app rendered inside Colab; zero visible code |
| Distribution | Public GitHub repo + "Open in Colab" badge |
| Single-photo engine | Hunyuan3D-2mini-Turbo, shape generation only (no textures) |
| Multi-photo engine | Hunyuan3D-2mv (multiview), offered as optional "advanced" mode |
| Output | Binary STL (+ GLB preview), scaled presets, optional flat base (default ON) |
| Language | All end-user-facing text (notebook prose, UI copy, errors, README) in neutral/professional Spanish |

## Architecture

One self-contained notebook, `modelo3d.ipynb`. Everything lives inside it so the
end user opens exactly one file. Three cells:

1. **Intro (markdown, ES):** what it does, requirements (Google account + photo),
   expected durations, that it is free, warning that sessions expire and files
   must be downloaded.
2. **Setup (code):** verifies GPU runtime; installs dependencies; downloads
   model weights (~2 GB); reports progress at each step.
3. **App (code):** constructs and renders the Gradio application inline.

### Processing Pipeline

```
photo(s)
  → input validation
  → background removal (rembg via Hunyuan3D BackgroundRemover)
  → shape generation
      · ≥2 valid views → Hunyuan3D-2mv
      · otherwise      → Hunyuan3D-2mini-Turbo
  → mesh repair (remove degenerate/duplicate faces, fill holes,
    enforce manifold/watertight)
  → normalization (upright orientation, resting on Z=0 plane,
    scale to target height)
  → optional flat base union (default ON)
  → export binary STL + GLB preview
```

Textures are intentionally skipped: FDM printing consumes geometry only.

### Size Presets

`10 cm` (default), `15 cm`, custom millimeters.

## End-User Flow

1. Click the "Open in Colab" badge in the repo README.
2. Enable the T4 GPU runtime (if skipped, the notebook blocks with exact
   click-path instructions: *Entorno de ejecución → Cambiar tipo de entorno de
   ejecución → T4 GPU*).
3. Run the setup cell (~5–8 min first run; progress messages explain each step).
4. In the app: upload photo(s) → pick mode ("Una foto" default / "Varias fotos")
   → pick size preset → press **Generar**.
5. Watch stage progress; receive rotating 3D preview; click **Descargar STL**;
   open the STL in Bambu Studio or PrusaSlicer.

### Multiview (Advanced) Mode

- Three labeled slots: Frente / Perfil / Espalda (left profile).
- Integrated guide: shoot all shots at the same distance and camera height,
  object centered identically in each frame; fixed azimuths only
  (~0°/90°/180°), no arbitrary angles.
- If fewer than two usable views arrive, the app falls back to the
  single-photo engine and explicitly tells the user it did so.

## Foolproofing Requirements

All conditions below produce friendly Spanish messages; the end user never sees
a raw stacktrace.

| Condition | Behavior |
|---|---|
| GPU runtime not enabled | Blocking message with exact click path |
| Missing/invalid/tiny image | Specific guidance per failure cause |
| Multiple objects suspected | Guidance: one object per photo, plain background |
| CUDA out-of-memory | One automatic retry with the engine's low-VRAM options (fewer inference steps / reduced octree resolution) + notice |
| Any other generation failure | Clear error + retry button |
| Session death | Intro warns: download results; rerun cells after restart |

Integrated photo-tips accordion in the app: good lighting, plain background,
single centered object, hands out of frame. Stage-by-stage progress feedback
("quitando fondo… generando geometría (~1 min)… reparando malla… listo").

Built-in **self-test**: bundled sample image runs the entire pipeline
(validation → repair → STL export) on demand before first real use, proving GPU,
dependencies, weights, and exporter all work.

## Out of Scope

- Texture/color generation or color printing
- Caching weights in Google Drive (future extension; adds auth friction)
- Arbitrary-angle multiview input
- Batch/production throughput; Colab Pro targeting
- Hosting outside Colab

## Repository Layout

```
modelo3d/
├── README.md        # ES: what it is, 3-step usage, photo guide, badge
├── modelo3d.ipynb   # the complete product (self-contained)
└── LICENSE          # MIT
```

No local Python package: the notebook must remain runnable from the single file.

## Testing Strategy

- **Notebook self-test** (in-product): sample image through full pipeline;
  success requires a watertight STL exported.
- **Manual Colab T4 protocol** before handoff:
  1. Single photo: hard-surface object.
  2. Single photo: pet/organic subject.
  3. Single photo: person/bust.
  4. Multiview happy path: three aligned views of one object.
  5. Multiview degraded path: only one usable view → confirm fallback works.
- **Local checks:** notebook JSON validity (nbformat), cell ordering, no
  absolute local paths.

## Risks and Notes

- **Free-tier variance:** daily GPU quotas and idle timeouts vary; setup reruns
  each session (accepted trade-off; stated in intro).
- **Multiview alignment sensitivity:** bad inputs can degrade output below
  single-photo quality; mitigated by integrated guide and automatic fallback.
- **Licensing:** Hunyuan3D family is commercially permissive; personal-use
  scenario assumed — re-verify license before any commercial use.
- **Model drift:** pin exact Hugging Face revision identifiers at implementation
  time so a silent upstream update cannot break the notebook.
