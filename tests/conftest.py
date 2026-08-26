import nbformat
import pytest

CORE_TAG = "core"
SELFTEST_TAG = "selftest"
NOTEBOOK = "modelo3d.ipynb"


def load_core_namespace():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ns: dict = {}
    for cell in nb.cells:
        tags = cell.get("metadata", {}).get("tags", [])
        if cell.cell_type == "code" and (
            CORE_TAG in tags or SELFTEST_TAG in tags
        ):
            exec(compile(cell.source, f"<cell:{cell.id}>", "exec"), ns)
    return ns


@pytest.fixture(scope="session")
def core():
    return load_core_namespace()
