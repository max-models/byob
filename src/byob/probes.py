"""Probe measurements and their harmonic interpolation onto a grid.

A :class:`Probe` fixes one quantity (``"density"``, ``"electron_temperature"``,
``"ion_temperature"``, ...) at one position. :func:`harmonic_fill` spreads the
probes of a quantity over the nodes of a uniform grid by solving Laplace's
equation with the probe nodes held fixed. Walls and the domain boundary are
zero-gradient, so values do not leak through electrodes, and the result never
overshoots the probe values. Positive quantities are interpolated in log space.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
import scope_profiler as sp

from byob._profiling import PROFILE_TAGS

__all__ = ["Probe", "UniformGrid", "harmonic_fill"]


@dataclass(frozen=True)
class Probe:
    """One measured or prescribed ``value`` of ``quantity`` at ``position`` (m)."""

    quantity: str
    position: tuple[float, ...]
    value: float


@dataclass(frozen=True)
class UniformGrid:
    """Nodes of a uniform tensor grid; the first and last node lie on the bounds.

    :meth:`from_simgeometry` takes the bounds and node counts of a
    :class:`simgeometry.grid.Grid`.
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


def _graph_laplacian(active: np.ndarray, spacing: np.ndarray) -> sparse.csr_matrix:
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
    adjacency = sparse.coo_matrix(
        (np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))),
        shape=(active.size, active.size),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sparse.diags(degree) - adjacency).tocsr()


@sp.profile("byob.harmonic_fill", tags=PROFILE_TAGS)
def harmonic_fill(
    grid: UniformGrid,
    probes: Sequence[Probe],
    *,
    solid: np.ndarray | None = None,
    log: bool = True,
) -> np.ndarray:
    """Interpolate ``probes`` over the nodes of ``grid`` by a Laplace solve.

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
        a = laplacian[free][:, free]
        b = -laplacian[free][:, fixed] @ target[fixed]
        result[free] = scipy.sparse.linalg.spsolve(a.tocsc(), b)
    result = result.reshape(shape)
    return np.exp(result) if log else result
