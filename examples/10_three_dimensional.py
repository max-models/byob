"""Three dimensions

A box chamber with a floating sphere, sampled in a mid-plane and along a line.

Nothing changes in 3D: the probes, the walls and :meth:`Background.sample` are the same, and a
fixed coordinate picks out a plane or a line, so plasma-plots draws the slice directly.
"""

import matplotlib.pyplot as plt
import plasma_plots  # noqa: F401
from simgeometry import Cuboid, Sphere

from byob import Background, Probe, UniformGrid, Wall

grid = UniformGrid((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (25, 25, 25))
walls = [
    Wall(Cuboid((0.05, 0.05, -0.005), (0.2, 0.2, 0.02))),  # floor at z = 0.005
    Wall(Sphere((0.05, 0.05, 0.05), 0.015)),
]
probes = [
    Probe("density", (0.02, 0.02, 0.09), 2e17),
    Probe("density", (0.09, 0.09, 0.02), 2e16),
    Probe("electron_temperature", (0.05, 0.05, 0.09), 3.0),
    Probe("ion_temperature", (0.05, 0.05, 0.09), 0.5),
]
background = Background.build(grid, probes, walls)

plane = background.sample(y=0.05, resolution=200)
plane.density.plasma.plot.slice(x="x", y="z", equal_aspect=True)
plane.potential.plasma.plot.slice(x="x", y="z", equal_aspect=True, levels=24)
background.sample(x=0.05, y=0.05, resolution=1000).potential.plasma.plot.lineout(x="z")
plt.show()
