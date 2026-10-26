---
title: Quickstart
description: Build a plasma background from probes and walls, and sample it.
---

First, ensure that `byob` is [installed](/byob/getting-started/installation/).

## Probes, walls, background

A background is set by probe values of the density (m⁻³), the electron and ion
temperatures (eV) and, optionally, the potential (V), and by walls: any
[simgeometry](https://github.com/max-models/simgeometry) shape, floating or biased.

```python
import numpy as np
from simgeometry import Rectangle

from byob import Background, Probe, UniformGrid, Wall

grid = UniformGrid((0.0, 0.0), (0.05, 0.02), (51, 21))
wall = Wall(Rectangle((-0.01, 0.01), (0.02, 0.04)))  # floating, solid for x < 0
probes = [
    Probe("density", (0.025, 0.01), 1e17),
    Probe("electron_temperature", (0.025, 0.01), 2.0),
    Probe("ion_temperature", (0.025, 0.01), 0.5),
]
background = Background.build(grid, probes, [wall])
```

The probes are interpolated over the grid, the bulk potential follows from Boltzmann's
relation, and the Debye sheath and presheath at each wall follow from the local electron
temperature and density.

## For particle tracking

`evaluate` and `electric_field` take `(N, ndim)` positions, as a particle pusher has them:

```python
positions = np.array([[0.02, 0.01], [1e-5, 0.01]])
state = background.evaluate(positions)  # densities, temperatures, potential
field = background.electric_field(positions)  # (N, 2), V/m
```

`background.profile("electron_density")` is a
[plasmacoll](https://github.com/max-models/plasmacoll) `Profile`, so it can shape a
collision background.

## Sampling and plotting

`sample` evaluates everything on a tensor grid and returns an `xarray.Dataset`. A scalar fixes
an axis, so lines and planes come out directly, ready for
[plasma-plots](https://github.com/max-models/plasma-plots):

```python
import plasma_plots

maps = background.sample(resolution=200)
maps.potential.plasma.plot.slice(x="x", y="y")
maps.electric_field.plasma.plot.vector(x="x", y="y", stride=10)

line = background.sample(x=np.linspace(0, 1e-3, 400), y=0.01)
line.potential.plasma.plot.lineout(x="x")
```

The [full examples](/byob/full-examples/) go through many cases: temperature and mass scans,
biased electrodes, probe interpolation, apertures, 3D and ion tracking.

See the [API reference](/byob/api/byob/) for every module and function.

## Profiling

byob's coarse operations are [scope-profiler](https://github.com/max-models/scope-profiler)
regions (`byob.Background.build`, `byob.fill_probes`, ...), tagged with the opt-in tag `byob`: they are
off by default and cost a fraction of a microsecond per call. A code that profiles itself with
scope-profiler turns them on with the tag:

```python
import scope_profiler as sp

sp.setup(enable_tags=["byob"])
```

or without editing the code, with `SCOPE_PROFILER_ENABLE_TAGS=byob`,
`scope-profiler run --enable-tag byob` or `enable_tags = ["byob"]` in the
TOML configuration. New regions use `@sp.profile("byob.<name>", tags=PROFILE_TAGS)`
with `PROFILE_TAGS` from the package's `_profiling` module.
