"""Partitioned chamber

A dividing wall with a gap: each probe only reaches the regions connected to it.

The interpolation runs over the grid graph of plasma nodes, so values flow around
walls, not through them. Here a hot dense plasma on the left reaches the right chamber only
through the gap at the top, and the density changes most along that path.
"""

import matplotlib.pyplot as plt
import plasma_plots  # noqa: F401
from simgeometry import Rectangle

from byob import Background, Probe, UniformGrid, Wall

grid = UniformGrid((0.0, 0.0), (0.2, 0.1), (101, 51))
divider = Wall(Rectangle((0.1, 0.04), (0.006, 0.08)))  # gap above y = 0.08
probes = [
    Probe("density", (0.03, 0.02), 1e18),
    Probe("density", (0.17, 0.02), 1e16),
    Probe("electron_temperature", (0.03, 0.02), 5.0),
    Probe("electron_temperature", (0.17, 0.02), 1.0),
    Probe("ion_temperature", (0.05, 0.05), 0.5),
]
background = Background.build(grid, probes, [divider])
maps = background.sample(resolution=300)
for name in ("density", "electron_temperature", "potential"):
    maps[name].plasma.plot.slice(x="x", y="y", equal_aspect=True)
plt.show()
