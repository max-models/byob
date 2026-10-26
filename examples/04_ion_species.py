"""Ion mass

The floating sheath drop for hydrogen, deuterium, helium and caesium ions.

The floating wall must repel enough electrons to balance the ion flux, so the drop grows with
the logarithm of the ion to electron mass ratio: about $2.8\\,T_e$ for H$^+$ and $5.3\\,T_e$
for Cs$^+$ (with cold ions). Heavier ions also make the presheath slower, but its potential
drop, $T_e/2$, does not depend on the mass.
"""

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots  # noqa: F401
from plasmacoll.constants import ATOMIC_MASS
from simgeometry import Rectangle

from byob import Background, Probe, UniformGrid, Wall

grid = UniformGrid((0.0, 0.0), (0.05, 0.02), (51, 21))
wall = Wall(Rectangle((-0.01, 0.01), (0.02, 0.04)))
centre = (0.025, 0.01)
probes = [
    Probe("density", centre, 1e17),
    Probe("electron_temperature", centre, 2.0),
    Probe("ion_temperature", centre, 0.1),
]
species = {"H$^+$": 1.008, "D$^+$": 2.014, "He$^+$": 4.003, "Cs$^+$": 132.9}

fig, ax = plt.subplots(layout="constrained")
for label, mass in species.items():
    background = Background.build(grid, probes, [wall], ion_mass=mass * ATOMIC_MASS)
    line = background.sample(x=np.linspace(0, 1.5e-3, 500), y=0.01)
    line.potential.plasma.plot.lineout(x="x", ax=ax)
    ax.lines[-1].set_label(f"{label}: wall at {line.potential[0].item():.1f} V")
ax.legend(frameon=False)
ax.set_title(r"floating wall, $T_e$ = 2 eV")
plt.show()
