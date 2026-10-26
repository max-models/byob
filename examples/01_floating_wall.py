"""Floating wall

The sheath and presheath in front of a floating wall in a uniform hydrogen plasma.

One probe of each quantity gives a uniform bulk: $n = 10^{17}$ m$^{-3}$, $T_e = 2$ eV,
$T_i = 0.5$ eV. The wall at $x = 0$ floats, so the potential falls by $T_e/2$ across the
presheath and by $\\frac{T_e}{2}\\ln\\frac{m_i}{2\\pi m_e(1 + T_i/T_e)}$ across the
Child-Langmuir sheath. The electrons follow Boltzmann's relation, the ions conserve the Bohm
flux, so the ion density exceeds the electron density inside the sheath. The lineouts zoom on
the first millimetre, where the Debye length is about 33 µm.
"""

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots  # noqa: F401  (registers the .plasma accessor)
from simgeometry import Rectangle

from byob import Background, Probe, UniformGrid, Wall, debye_length

grid = UniformGrid((0.0, 0.0), (0.05, 0.02), (51, 21))
wall = Wall(Rectangle((-0.01, 0.01), (0.02, 0.04)))  # solid for x < 0
centre = (0.025, 0.01)
probes = [
    Probe("density", centre, 1e17),
    Probe("electron_temperature", centre, 2.0),
    Probe("ion_temperature", centre, 0.5),
]
background = Background.build(grid, probes, [wall])

line = background.sample(x=np.linspace(0, 1e-3, 400), y=0.01)
print(f"Debye length: {debye_length(1e17, 2.0) * 1e6:.1f} um")
print(f"wall potential: {line.potential[0].item():.2f} V")

fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
line.potential.plasma.plot.lineout(x="x", ax=axes[0, 0])
line.electron_density.plasma.plot.lineout(x="x", ax=axes[0, 1])
line.ion_density.plasma.plot.lineout(x="x", ax=axes[0, 1])
axes[0, 1].legend(["electrons", "ions"], frameon=False)
axes[0, 1].set_title("densities")
line.electric_field.sel(component="x").plasma.plot.lineout(x="x", ax=axes[1, 0])
axes[1, 0].set_yscale("symlog", linthresh=1e2)
(line.ion_density - line.electron_density).rename("space charge").plasma.plot.lineout(
    x="x", ax=axes[1, 1]
)
plt.show()
