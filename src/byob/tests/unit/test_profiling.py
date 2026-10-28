"""The opt-in scope-profiler regions: off unless the application enables the tag."""

from __future__ import annotations

import pytest
import scope_profiler as sp

from byob.main import main


@pytest.fixture(autouse=True)
def _fresh_profiler(monkeypatch):
    """Keep the profiling state of these tests out of the others."""
    monkeypatch.delenv("SCOPE_PROFILER_ENABLE_TAGS", raising=False)
    sp.reset()
    yield
    sp.reset()


def _region_names(**setup_kwargs) -> set[str]:
    sp.setup(output=None, **setup_kwargs)
    # an application region, recorded either way
    with sp.region("user.run"):
        main()
    results = sp.finalize(verbose=False, return_results=True)
    return set(results.region_names)


def test_regions_are_off_by_default() -> None:
    assert _region_names() == {"user.run"}


@pytest.mark.parametrize("tags", [["byob"], ["*"]])
def test_enabling_the_tag_records_the_regions(tags: list[str]) -> None:
    assert _region_names(enable_tags=tags) == {"byob.main", "user.run"}


def test_the_opt_in_tag_has_an_entry_point():
    """``scope-profiler tags`` finds the tag through the package's entry point."""
    from importlib.metadata import entry_points

    from byob._profiling import PROFILE_TAGS

    found = {
        entry_point.name: entry_point.value
        for entry_point in entry_points(group="scope_profiler.opt_in_tags")
    }
    assert found.get("byob") == "byob"
    assert "byob" in PROFILE_TAGS
