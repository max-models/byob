"""Biased probe tip

A cylindrical probe tip at $-60$ V in a plasma with a density gradient.

The tip is a circle in this 2D cut. Its sheath surrounds it with a width set by the local
density and temperature; the electric field (arrows) points into the tip within the sheath
and is weak in the bulk, where only the density gradient drives it through Boltzmann's
relation. The lineout crosses the tip along $y$.
"""

import matplotlib.pyplot as plt
import plasma_plots  # noqa: F401
from simgeometry import Circle

from byob import Background, Probe, UniformGrid, Wall

grid = UniformGrid((0.0, 0.0), (0.02, 0.02), (81, 81))
tip = Wall(Circle((0.01, 0.01), 0.0015), kind="biased", bias=-60.0)
probes = [
    Probe("density", (0.0, 0.01), 2e16),
    Probe("density", (0.02, 0.01), 2e15),
    Probe("electron_temperature", (0.005, 0.005), 4.0),
    Probe("ion_temperature", (0.005, 0.005), 0.3),
]
background = Background.build(grid, probes, [tip])
maps = background.sample(resolution=240)
maps.potential.plasma.plot.slice(x="x", y="y", equal_aspect=True, levels=24)

arrows = background.sample(resolution=41)
result = arrows.electric_field.plasma.plot.vector(x="x", y="y")
result.ax.set_aspect("equal")
result.ax.set_title("electric field")

maps.ion_density.sel(x=0.01, method="nearest").plasma.plot.lineout(x="y")
plt.show()
