"""Interpolating density probes

Five density probes spread over a chamber by biharmonic interpolation, with walls on two sides.

Between the probes, $\\ln n$ minimises its squared Laplacian (a thin-plate spline on the grid),
with the probes held fixed and zero gradient at walls and domain edges. The result passes
smoothly through every probe; it can overshoot them slightly, which
``method="harmonic"`` avoids at the price of a cusp at each probe. The bulk potential follows from Boltzmann's relation,
$\\phi = T_e\\ln(n/n_\\mathrm{max})$, and the sheaths are added at the walls. The white points
mark the probes.
"""

import matplotlib.pyplot as plt
import plasma_plots  # noqa: F401
from simgeometry import Rectangle

from byob import Background, Probe, UniformGrid, Wall

grid = UniformGrid((0.0, 0.0), (0.2, 0.1), (101, 51))
walls = [
    Wall(Rectangle((0.1, 0.0), (0.2, 0.01))),  # bottom
    Wall(Rectangle((0.2, 0.05), (0.01, 0.1))),  # right
]
density = {
    (0.03, 0.08): 3e17,
    (0.05, 0.03): 1e17,
    (0.10, 0.06): 5e17,
    (0.15, 0.02): 4e16,
    (0.17, 0.08): 1.5e17,
}
probes = [Probe("density", p, n) for p, n in density.items()]
probes += [
    Probe("electron_temperature", (0.1, 0.05), 2.0),
    Probe("ion_temperature", (0.1, 0.05), 0.5),
]
background = Background.build(grid, probes, walls)
maps = background.sample(resolution=300)
points = {"probes": tuple(zip(*density, strict=True))}

maps.density.plasma.plot.slice(
    x="x", y="y", equal_aspect=True, overlays={"points": points}
)
maps.potential.plasma.plot.slice(
    x="x", y="y", equal_aspect=True, levels=20, overlays={"points": points}
)
maps.density.sel(y=0.06, method="nearest").plasma.plot.lineout(x="x")
plt.show()
