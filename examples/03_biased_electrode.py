"""Biased electrode

A negatively biased wall: the sheath widens following the Child-Langmuir law.

The wall potential is set instead of floating. Below the sheath edge potential the drop is
$\\Delta\\phi = \\phi_\\mathrm{edge} - V_\\mathrm{bias}$ and the width
$s = \\frac{\\sqrt 2}{3}\\lambda_D (2e\\Delta\\phi/T_e)^{3/4}$. A bias above the plasma
potential would draw an electron sheath, which byob does not model: the wall then has only the
presheath.
"""

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots  # noqa: F401
from simgeometry import Rectangle

from byob import Background, Probe, UniformGrid, Wall, sheath_width

grid = UniformGrid((0.0, 0.0), (0.05, 0.02), (51, 21))
shape = Rectangle((-0.01, 0.01), (0.02, 0.04))
centre = (0.025, 0.01)
probes = [
    Probe("density", centre, 1e17),
    Probe("electron_temperature", centre, 3.0),
    Probe("ion_temperature", centre, 0.5),
]

biases = np.array([-10.0, -50.0, -200.0, -1000.0])
fig, (ax_phi, ax_n) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for bias in biases:
    background = Background.build(grid, probes, [Wall(shape, "biased", bias)])
    line = background.sample(x=np.linspace(0, 6e-3, 800), y=0.01)
    line.potential.plasma.plot.lineout(x="x", ax=ax_phi)
    line.electron_density.plasma.plot.lineout(x="x", ax=ax_n)
    ax_phi.lines[-1].set_label(f"{bias:g} V")
ax_phi.legend(frameon=False, title="bias")
ax_n.set_yscale("log")
ax_n.set_title("electron density")

edge = 1e17 * np.exp(-0.5)
drops = np.geomspace(1, 3000, 100)
fig, ax = plt.subplots(layout="constrained")
ax.loglog(drops, sheath_width(drops, edge, 3.0) * 1e3)
ax.set(
    xlabel="sheath drop (V)",
    ylabel="sheath width (mm)",
    title=r"Child-Langmuir: $s \propto \Delta\phi^{3/4}$",
)
plt.show()
