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
    assert "friendly_error(exc)" in src
    assert "print(traceback" not in src
