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
