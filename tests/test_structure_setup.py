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
    shas = re.findall(r'"(mini|mv)":\s*"([0-9a-f]{40})"', src)
    assert len(shas) == 2
    for key, sha in shas:
        assert SHA_RE.match(sha), f"not a pinned sha: {sha}"
