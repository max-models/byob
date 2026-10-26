"""The scope-profiler tag of byob's profiling regions.

The regions are opt-in: :mod:`byob` registers the tag on import, so they
record nothing unless the application enables it, e.g. with
``scope_profiler.setup(enable_tags=["byob"])``.
"""

#: Tags of every byob region; the first is the opt-in tag.
PROFILE_TAGS = ("byob",)
