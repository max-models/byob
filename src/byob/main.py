"""Main module of the python package."""

import scope_profiler as sp

from byob._profiling import PROFILE_TAGS


@sp.profile("byob.main", tags=PROFILE_TAGS)
def main() -> None:
    """Print a greeting; the entry point of the ``byob`` command."""
    print("Hello, world")
