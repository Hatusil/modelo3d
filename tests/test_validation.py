import numpy as np
import pytest
from PIL import Image


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


def test_friendly_error_passes_repair_message(core):
    exc = ValueError("El modelo salió con agujeros que no pudimos reparar. Probá generar de nuevo con otra foto.")
    msg = core["friendly_error"](exc)
    assert "agujeros" in msg
