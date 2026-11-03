"""Probe measurements and their interpolation onto a grid.

A `Probe` fixes one quantity (``"density"``, ``"electron_temperature"``,
``"ion_temperature"``, ...) at one position. `fill_probes` spreads the
probes of a quantity over the nodes of a uniform grid by solving a
(bi)harmonic equation on the grid graph with the probe nodes held fixed. Walls
and the domain boundary are zero-gradient, so values flow around electrodes
instead of through them. Positive quantities are interpolated in log space.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
import scope_profiler as sp
from methodcard import ConvergenceProblem, method

from byob._profiling import PROFILE_TAGS

__all__ = ["Probe", "UniformGrid", "fill_probes"]


@dataclass(frozen=True)
class Probe:
    """One measured or prescribed ``value`` of ``quantity`` at ``position`` (m)."""

    quantity: str
    position: tuple[float, ...]
    value: float


@dataclass(frozen=True)
@method(
    equations={
        "nodes": r"x_{d,k} = x_{d,\mathrm{lo}} + k\, h_d, \qquad h_d = \frac{x_{d,\mathrm{hi}} - x_{d,\mathrm{lo}}}{N_d - 1}, \qquad k = 0, \dots, N_d - 1",
        "nearest_node": r"k_d(\mathbf x) = \min\left(\max\left(\operatorname{rint}\frac{x_d - x_{d,\mathrm{lo}}}{h_d},\, 0\right),\, N_d - 1\right)",
    },
    assumptions=(
        r"a tensor grid; the first and last node of each axis lie on the bounds",
    ),
)
class UniformGrid:
    """Nodes of a uniform tensor grid; the first and last node lie on the bounds.

    `from_simgeometry` takes the bounds and node counts of a
    `simgeometry.grid.Grid`.
    """

    lower_bound: tuple[float, ...]
    upper_bound: tuple[float, ...]
    num_gridpoints: tuple[int, ...]

    @classmethod
    def from_simgeometry(cls, grid) -> UniformGrid:
        """Return the uniform grid with the bounds and node counts of ``grid``."""
        return cls(
            tuple(float(v) for v in grid.lower_bound),
            tuple(float(v) for v in grid.upper_bound),
            tuple(int(v) for v in grid.num_gridpoints),
        )

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return len(self.num_gridpoints)

    @property
    def axes(self) -> tuple[np.ndarray, ...]:
        """Return the node coordinates along each axis."""
        return tuple(
            np.linspace(lo, hi, n)
            for lo, hi, n in zip(
                self.lower_bound, self.upper_bound, self.num_gridpoints, strict=True
            )
        )

    @property
    def spacing(self) -> np.ndarray:
        """Return the node spacing along each axis."""
        upper, lower = np.array(self.upper_bound), np.array(self.lower_bound)
        return (upper - lower) / (np.array(self.num_gridpoints) - 1)

    def points(self) -> np.ndarray:
        """Return all node positions, component-major ``(ndim, n)`` in C order."""
        mesh = np.meshgrid(*self.axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh])

    def nearest_node(self, position: Sequence[float]) -> tuple[int, ...]:
        """Return the index of the node nearest to ``position``."""
        index = np.rint(
            (np.asarray(position) - np.array(self.lower_bound)) / self.spacing
        ).astype(int)
        return tuple(np.clip(index, 0, np.array(self.num_gridpoints) - 1).tolist())


def _graph_laplacian(active: np.ndarray, spacing: np.ndarray) -> sparse.csr_array:
    """Return the Laplacian of the grid graph that links neighbouring active nodes.

    A link along an axis weighs ``1/spacing**2``. Links to inactive nodes and
    across the domain boundary are dropped, a zero-gradient condition there.
    """
    shape = active.shape
    index = np.arange(active.size).reshape(shape)
    rows, cols, weights = [], [], []
    for axis in range(active.ndim):
        lo = [slice(None)] * active.ndim
        hi = [slice(None)] * active.ndim
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        both = active[tuple(lo)] & active[tuple(hi)]
        rows.append(index[tuple(lo)][both])
        cols.append(index[tuple(hi)][both])
        weights.append(np.full(int(both.sum()), spacing[axis] ** -2))
    i, j, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(weights)
    adjacency = sparse.coo_array(
        (np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))),
        shape=(active.size, active.size),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return sparse.csr_array(sparse.diags_array(degree) - adjacency)


def _fill_probes_problem() -> ConvergenceProblem:
    """The biharmonic fill of four probes in 1D against its limit, a cubic spline.

    In 1D, minimising the squared graph Laplacian with the probe nodes fixed tends
    to the cubic spline through the probes with zero slope at the domain ends
    (the zero-gradient condition of the dropped links).
    """
    from scipy.interpolate import CubicSpline

    positions, values = (0.0, 0.25, 0.5, 1.0), (1.0, 3.0, 2.0, 0.5)
    spline = CubicSpline(positions, values, bc_type="clamped")
    probes = [Probe("q", (x,), v) for x, v in zip(positions, values, strict=True)]

    def error(spacing: float) -> float:
        grid = UniformGrid((0.0,), (1.0,), (round(1.0 / spacing) + 1,))
        filled = fill_probes(grid, probes, log=False)
        return float(np.max(np.abs(filled - spline(grid.axes[0]))))

    return ConvergenceProblem(error, steps=[1 / 64, 1 / 128, 1 / 256, 1 / 512])


@method(
    equations={
        "log": r"u_k = \ln Q_k, \qquad Q_k = e^{u_k}",
        "harmonic": r"(L u)_k = 0",
        "biharmonic": r"(L^{\mathsf T} L\, u)_k = 0 \quad\Longleftrightarrow\quad \min_{u_U} \lVert L u \rVert^2",
        "laplacian": r"L_{kl} = -\frac{1}{h_d^2} \text{ for active neighbours } k, l \text{ along axis } d, \qquad L_{kk} = -\sum_{l \ne k} L_{kl}",
        "system": r"M_{UU}\, u_U = -M_{UF}\, u_F, \qquad M = L \text{ or } L^{\mathsf T} L",
    },
    order=1,
    assumptions=(
        r"$F$ the probe nodes (each probe snapped to its nearest node; the later of two on one node wins), $U$ the free nodes: active and connected to a probe",
        r"links to wall nodes and across the domain boundary dropped: zero gradient there, a mirror node with the same value",
        r"harmonic: a weighted mean of the neighbours, no overshoot (discrete maximum principle), a logarithmic cusp at a point probe in 2D and 3D",
        r"biharmonic (default): a discrete thin-plate spline, smooth through the probes, small overshoots possible between them",
        r"`log` (default) interpolates $\ln Q$, so positive quantities stay positive; wall nodes and regions without a probe are NaN",
        r"one sparse direct solve (`scipy.sparse.linalg.spsolve`) per quantity",
        r"the order is that of the biharmonic fill in 1D, against the cubic spline through the probes with zero slope at the domain ends",
    ),
    problem=_fill_probes_problem,
)
@sp.profile("byob.fill_probes", tags=PROFILE_TAGS)
def fill_probes(
    grid: UniformGrid,
    probes: Sequence[Probe],
    *,
    solid: np.ndarray | None = None,
    log: bool = True,
    method: Literal["biharmonic", "harmonic"] = "biharmonic",
) -> np.ndarray:
    """Interpolate ``probes`` over the nodes of ``grid``.

    ``"biharmonic"`` minimises the squared Laplacian, a discrete thin-plate
    spline: smooth through the probes, but it may overshoot them a little
    between probes. ``"harmonic"`` solves Laplace's equation: it never
    overshoots, but in 2D and 3D it has a logarithmic cusp at every probe.

    ``solid`` (boolean, grid-shaped) marks nodes inside walls; they are left out
    of the solve and set to NaN. With ``log`` the logarithm of the values is
    interpolated, which keeps positive quantities positive. A single probe gives
    a uniform field. Every connected region of non-solid nodes needs a probe,
    otherwise it is set to NaN.
    """
    if not probes:
        raise ValueError("need at least one probe")
    shape = grid.num_gridpoints
    active = np.ones(shape, dtype=bool) if solid is None else ~np.asarray(solid)
    values = np.array([p.value for p in probes], dtype=float)
    if log:
        if np.any(values <= 0):
            raise ValueError("log interpolation needs positive probe values")
        values = np.log(values)

    fixed = np.zeros(active.size, dtype=bool)
    target = np.zeros(active.size)
    for probe, value in zip(probes, values, strict=True):
        flat = np.ravel_multi_index(grid.nearest_node(probe.position), shape)
        if not active.ravel()[flat]:
            raise ValueError(f"probe at {probe.position} lies inside a wall")
        fixed[flat], target[flat] = True, value

    laplacian = _graph_laplacian(active, grid.spacing)
    # Nodes linked to no probe have a singular block; find and drop them.
    _, labels = scipy.sparse.csgraph.connected_components(
        laplacian != 0, directed=False
    )
    reached = np.isin(labels, labels[fixed]) & active.ravel()
    free = reached & ~fixed

    result = np.full(active.size, np.nan)
    result[fixed] = target[fixed]
    if free.any():
        operator = laplacian if method == "harmonic" else laplacian.T @ laplacian
        a = operator[free][:, free]
        b = -operator[free][:, fixed] @ target[fixed]
        result[free] = scipy.sparse.linalg.spsolve(a.tocsc(), b)
    result = result.reshape(shape)
    return np.exp(result) if log else result
