"""Electron temperature gradient

A hot driver region next to a cold expansion region, as in a negative ion source.

Two electron temperature probes, 10 eV in the driver and 1 eV near the extraction side, set a
smooth temperature profile. The density is uniform here, so the bulk potential is flat; the
temperature still changes the sheath at each wall: the floating drop scales with the local
$T_e$, so the left wall sits much lower than the right one.
"""

import matplotlib.pyplot as plt
import plasma_plots  # noqa: F401
from simgeometry import Rectangle

from byob import Background, Probe, UniformGrid, Wall

grid = UniformGrid((0.0, 0.0), (0.3, 0.1), (121, 41))
walls = [
    Wall(Rectangle((-0.005, 0.05), (0.02, 0.2))),  # driver back plate, x < 0.005
    Wall(Rectangle((0.305, 0.05), (0.02, 0.2))),  # plasma grid, x > 0.295
]
probes = [
    Probe("density", (0.15, 0.05), 1e17),
    Probe("electron_temperature", (0.05, 0.05), 10.0),
    Probe("electron_temperature", (0.25, 0.05), 1.0),
    Probe("ion_temperature", (0.15, 0.05), 0.8),
]
background = Background.build(grid, probes, walls)
maps = background.sample(resolution=240)
maps.electron_temperature.plasma.plot.slice(x="x", y="y", equal_aspect=True)

line = background.sample(y=0.05, resolution=3000)
fig, (ax_t, ax_phi) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
line.electron_temperature.plasma.plot.lineout(x="x", ax=ax_t)
line.potential.plasma.plot.lineout(x="x", ax=ax_phi)
plt.show()
