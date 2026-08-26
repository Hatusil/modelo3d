import nbformat


def test_selftest_cell_runs_full_chain():
    nb = nbformat.read("modelo3d.ipynb", as_version=4)
    src = next(c.source for c in nb.cells
               if "selftest" in c.get("metadata", {}).get("tags", []))
    assert "make_sample_image" in src
    assert "verify_stl" in src
    assert "SELF TEST" in src  # bilingual summary marker
