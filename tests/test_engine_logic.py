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


def test_run_pipeline_glue_exists(core):
    assert callable(core["run_pipeline"])
