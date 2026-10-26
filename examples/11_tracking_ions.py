"""Tracking ions

What the background is for: pushing particles through its electric field.

Hydrogen ions start at rest in the bulk and are integrated with a leapfrog scheme, calling
:meth:`Background.electric_field` at their positions every step. They drift slowly down the
presheath and are then accelerated through the sheath onto the biased electrode below. The
trajectories are drawn over the potential map; the second figure shows that ions reach the
sheath edge near the Bohm speed $\\sqrt{T_e/m_i}$.
"""

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots  # noqa: F401
from plasmacoll.constants import ELEMENTARY_CHARGE, PROTON_MASS
from simgeometry import Rectangle

from byob import Background, Probe, UniformGrid, Wall

te = 2.0
grid = UniformGrid((0.0, 0.0), (0.004, 0.002), (81, 41))
electrode = Wall(Rectangle((0.002, -0.001), (0.006, 0.002)), kind="biased", bias=-30.0)
probes = [
    Probe("density", (0.002, 0.002), 1.2e18),
    Probe("density", (0.002, 0.0008), 1e18),
    Probe("electron_temperature", (0.002, 0.0015), te),
    Probe("ion_temperature", (0.002, 0.0015), 0.1),
]
background = Background.build(grid, probes, [electrode], presheath_debye_lengths=50.0)

n = 6
position = np.column_stack([np.linspace(0.0005, 0.0035, n), np.full(n, 0.0016)])
velocity = np.zeros_like(position)
dt = 2e-10
q_over_m = ELEMENTARY_CHARGE / PROTON_MASS
paths, speeds = [position.copy()], [np.zeros(n)]
alive = np.ones(n, dtype=bool)
for _ in range(20000):
    velocity[alive] += q_over_m * background.electric_field(position[alive]) * dt
    position[alive] += velocity[alive] * dt
    alive &= position[:, 1] > 0.0
    paths.append(position.copy())
    speeds.append(np.linalg.norm(velocity, axis=1))
    if not alive.any():
        break
paths = np.array(paths)

maps = background.sample(resolution=240)
lines = {f"ion {i}": (paths[:, i, 0], paths[:, i, 1]) for i in range(n)}
maps.potential.plasma.plot.slice(
    x="x", y="y", equal_aspect=True, overlays={"lines": lines}
)

bohm = np.sqrt(te * ELEMENTARY_CHARGE / PROTON_MASS)
fig, ax = plt.subplots(layout="constrained")
ax.plot(paths[:, 0, 1] * 1e3, np.array(speeds)[:, 0] / bohm)
ax.axhline(1, color="k", ls=":")
ax.set(
    xlabel="height y (mm)",
    ylabel=r"speed / $c_s$",
    title="ion 0 falling onto the electrode",
)
ax.invert_xaxis()
plt.show()
