"""The animated figure at the top of the docs home page: ``docs/src/assets/hero.svg``.

A Debye sheath in front of a biased electrode, built by byob from a few probes: equipotentials of
the background bend around the electrode, hydrogen ions start at rest in the bulk, drift down the
presheath and are accelerated onto the electrode, while electrons coming down are turned back by
the sheath. The particles are pushed through :meth:`byob.Background.electric_field` with a
leapfrog scheme (a few seconds). Rerun with ``make docs-hero`` or ``python docs/hero.py``.
"""

from pathlib import Path

import contourpy
import numpy as np
from plasma_plots.hero import HeroFigure
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, PROTON_MASS
from simgeometry import Rectangle

from byob import Background, Probe, UniformGrid, Wall

OUTPUT = Path(__file__).parent / "src" / "assets" / "hero.svg"
WIDTH, HEIGHT = 0.008, 0.003
ELECTRODE = (0.0034, 0.0005)  # left edge and top of the electrode


def push(background, position, velocity, q_over_m, dt, steps, stop, stride):
    """Leapfrog until ``stop(position, velocity)`` holds for all; every ``stride``-th step."""
    paths = [position.copy()]
    alive = np.ones(len(position), dtype=bool)
    for _ in range(steps):
        velocity[alive] += q_over_m * background.electric_field(position[alive]) * dt
        position[alive] += velocity[alive] * dt
        alive &= ~stop(position, velocity)
        paths.append(position.copy())
        if not alive.any():
            break
    paths = np.array(paths)
    return paths[np.unique(np.r_[0 : len(paths) : stride, len(paths) - 1])]


grid = UniformGrid((0.0, 0.0), (WIDTH, HEIGHT), (161, 61))
electrode = Wall(
    Rectangle(
        ((ELECTRODE[0] + WIDTH + 0.001) / 2, (ELECTRODE[1] - 0.001) / 2),
        (WIDTH + 0.001 - ELECTRODE[0], ELECTRODE[1] + 0.001),
    ),
    kind="biased",
    bias=-30.0,
)
probes = [
    Probe("density", (0.004, 0.003), 1.5e16),
    Probe("density", (0.004, 0.0012), 1e16),
    Probe("electron_temperature", (0.004, 0.002), 2.0),
    Probe("ion_temperature", (0.004, 0.002), 0.1),
]
background = Background.build(grid, probes, [electrode], presheath_debye_lengths=10.0)

fig = HeroFigure(xlim=(0.0, WIDTH), ylim=(0.0, HEIGHT))

maps = background.sample(resolution=200)
phi = maps.potential.values.T
generator = contourpy.contour_generator(maps.x.values, maps.y.values, phi)
for level in np.linspace(np.nanmin(phi) + 1.0, np.nanmax(phi) - 0.05, 14):
    for line in generator.lines(level):
        if len(line) > 3:
            fig.line(line[::4, 0], line[::4, 1], tone="muted", width=1.0, opacity=0.45)
fig.rect(
    ELECTRODE[0], -0.001, WIDTH + 0.001, ELECTRODE[1], tone="low", outline="accent"
)

# ions from rest in the bulk, falling onto the electrode
n_ions = 9
start = np.column_stack([np.linspace(0.0038, 0.0076, n_ions), np.full(n_ions, 0.0026)])
ions = push(
    background,
    start.copy(),
    np.zeros_like(start),
    ELEMENTARY_CHARGE / PROTON_MASS,
    4e-10,
    80000,
    lambda x, v: x[:, 1] < ELECTRODE[1],
    60,
)
for i in range(n_ions):
    x, y = ions[:, i, 0], ions[:, i, 1]
    fig.line(x, y, tone="accent", width=1.2, opacity=0.5)
    for k in range(2):
        fig.tracer(
            x, y=y, duration=6.0, offset=6.0 * (k / 2 + i / n_ions), r=3.5, tone="high"
        )

# electrons of 8 eV coming down and turned back by the sheath
n_electrons = 7
angle = np.linspace(-0.5, 0.5, n_electrons)
speed = np.sqrt(2 * 8.0 * ELEMENTARY_CHARGE / ELECTRON_MASS)
start = np.column_stack(
    [np.linspace(0.0042, 0.0074, n_electrons), np.full(n_electrons, 0.0022)]
)
velocity = speed * np.column_stack([np.sin(angle), -np.cos(angle)])
electrons = push(
    background,
    start.copy(),
    velocity,
    -ELEMENTARY_CHARGE / ELECTRON_MASS,
    2e-13,
    20000,
    lambda x, v: (v[:, 1] > 0) & (x[:, 1] > 0.0022) | (x[:, 1] < ELECTRODE[1]),
    40,
)
for i in range(n_electrons):
    x, y = electrons[:, i, 0], electrons[:, i, 1]
    fig.line(x, y, tone="muted", width=1.0, opacity=0.6, dash="3 4")
    fig.tracer(x, y=y, duration=4.0, offset=4.0 * i / n_electrons, r=2.5, tone="accent")

fig.save(OUTPUT)
print(f"{OUTPUT}: {OUTPUT.stat().st_size / 1e3:.0f} kB")
