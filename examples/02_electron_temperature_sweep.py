"""Sheath versus electron temperature

How the floating sheath scales with $T_e$: the drop grows linearly, the width with the Debye length.

At fixed density the Debye length grows as $\\sqrt{T_e}$ and the floating drop as $T_e$, so
hotter plasmas have wider, deeper sheaths. Plotted against the distance in Debye lengths and
the potential in units of $T_e$, the curves collapse onto one, since the sheath only depends on
$T_e$ through those scales (and weakly on $T_i/T_e$, held fixed here).
"""

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots  # noqa: F401
from simgeometry import Rectangle

from byob import Background, Probe, UniformGrid, Wall, debye_length

grid = UniformGrid((0.0, 0.0), (0.05, 0.02), (51, 21))
wall = Wall(Rectangle((-0.01, 0.01), (0.02, 0.04)))
centre = (0.025, 0.01)
density = 1e17

fig, (ax_si, ax_norm) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for te in (1.0, 3.0, 10.0, 30.0):
    probes = [
        Probe("density", centre, density),
        Probe("electron_temperature", centre, te),
        Probe("ion_temperature", centre, te / 4),
    ]
    background = Background.build(grid, probes, [wall])
    line = background.sample(x=np.linspace(0, 3e-3, 600), y=0.01)
    line.potential.plasma.plot.lineout(x="x", ax=ax_si)

    debye = debye_length(density, te)
    scaled = (line.potential / te).assign_coords(x=line.x / debye)
    scaled.rename("potential / Te").plasma.plot.lineout(x="x", ax=ax_norm)
    ax_si.lines[-1].set_label(f"$T_e$ = {te:g} eV")

ax_si.legend(frameon=False)
ax_norm.set(xlim=(0, 60), xlabel=r"distance / $\lambda_D$", ylabel=r"$e\phi / T_e$")
ax_norm.set_title("normalised: the curves collapse")
plt.show()
