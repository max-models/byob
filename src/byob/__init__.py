"""Build your own background: static plasma fields with sheaths for particle tracking."""

from importlib.metadata import PackageNotFoundError, version

import scope_profiler as sp

from byob._profiling import PROFILE_TAGS
from byob.background import Background, FieldProfile
from byob.probes import Probe, UniformGrid, harmonic_fill
from byob.sheath import Wall, debye_length, sheath_drop, sheath_width

__all__ = [
    "Background",
    "FieldProfile",
    "Probe",
    "UniformGrid",
    "Wall",
    "debye_length",
    "harmonic_fill",
    "sheath_drop",
    "sheath_width",
]

try:
    __version__ = version("byob")
except PackageNotFoundError:  # not installed, e.g. run from a source tree
    __version__ = "0.0.0+unknown"

# The decorated functions pick the tag up, so registering after the imports is in time.
sp.register_opt_in_tag(*PROFILE_TAGS)
