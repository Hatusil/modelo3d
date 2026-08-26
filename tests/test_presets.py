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
