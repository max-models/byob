"""Extraction aperture

A plasma grid with one aperture: the sheath follows the grid and bends into the hole.

The grid is two rectangles leaving an aperture of 8 mm. The plasma fills the left side, so the
density probe sits there; across the aperture the probes give a low density, which is
interpolated through the opening only. Each point takes the sheath of the wall that pulls
its potential lowest: the nearest one away from the edges. That is a good model where the wall
is much larger than the sheath and a rough one at the aperture edges.
"""

import matplotlib.pyplot as plt
import plasma_plots  # noqa: F401
from simgeometry import Rectangle

from byob import Background, Probe, UniformGrid, Wall

grid = UniformGrid((0.0, -0.02), (0.03, 0.02), (121, 161))
upper = Wall(Rectangle((0.02, 0.012), (0.002, 0.016)), kind="biased", bias=0.0)
lower = Wall(Rectangle((0.02, -0.012), (0.002, 0.016)), kind="biased", bias=0.0)
probes = [
    Probe("density", (0.002, 0.0), 1e17),
    Probe("density", (0.029, 0.0), 1e15),
    Probe("electron_temperature", (0.002, 0.0), 2.0),
    Probe("ion_temperature", (0.002, 0.0), 0.8),
    Probe("potential", (0.002, 0.0), 20.0),  # plasma potential above the grid
]
background = Background.build(grid, probes, [upper, lower])
maps = background.sample(resolution=320)
maps.density.plasma.plot.slice(x="x", y="y", equal_aspect=True)
maps.potential.plasma.plot.slice(x="x", y="y", equal_aspect=True, levels=24)
maps.ion_density.sel(y=0.008, method="nearest").plasma.plot.lineout(x="x")
plt.show()
