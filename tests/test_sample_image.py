import numpy as np


def test_sample_image_deterministic_and_usable(core):
    a = np.array(core["make_sample_image"]())
    b = np.array(core["make_sample_image"]())
    assert (a == b).all()
    assert a.shape[:2] == (768, 768)
    assert a.std() > 10  # enough contrast for background removal
