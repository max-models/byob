"""A static plasma background for particle tracking, built from probes and walls.

:meth:`Background.build` interpolates the probed density and temperatures over a
grid (:func:`~byob.probes.harmonic_fill`) and sets the bulk potential from the
Boltzmann relation ``phi = phi_ref + Te ln(n / n_ref)``. Sheaths and presheaths
(:mod:`byob.sheath`) are added analytically at particle positions, so they stay
sharp however coarse the grid.

Positions are ``(N, ndim)`` in m, densities in m^-3, temperatures in eV,
potentials in V and electric fields in V/m.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scope_profiler as sp
from plasmacoll.constants import PROTON_MASS
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from byob._profiling import PROFILE_TAGS
from byob.probes import Probe, UniformGrid, harmonic_fill
from byob.sheath import PlasmaState, Wall, debye_length, wall_layer

__all__ = ["Background", "FieldProfile"]

_BULK = ("density", "electron_temperature", "ion_temperature", "potential")


def _fill_solid(values: np.ndarray) -> np.ndarray:
    """Return ``values`` with NaN nodes set to the nearest finite node."""
    missing = np.isnan(values)
    if not missing.any():
        return values
    _, index = distance_transform_edt(missing, return_indices=True)
    return values[tuple(index)]


@dataclass(frozen=True)
class FieldProfile:
    """A background quantity over a reference value: a :class:`plasmacoll` Profile.

    ``positions`` are ``(N, ndim)``, as for the markers of plasmacoll.
    """

    background: Background
    quantity: str
    reference: float

    @property
    def max_factor(self) -> float:
        """Return the largest factor, bounded by the bulk maximum on the grid."""
        return float(
            np.nanmax(self.background._grid_values(self.quantity)) / self.reference
        )

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Return the factor at ``positions``."""
        return self.background.evaluate(positions)[self.quantity] / self.reference


@dataclass
class Background:
    """Bulk fields on a grid, with sheaths at ``walls`` added at evaluation."""

    grid: UniformGrid
    bulk: dict[str, np.ndarray]
    walls: Sequence[Wall] = ()
    ion_mass: float = PROTON_MASS
    presheath_debye_lengths: float = 20.0
    _interpolators: dict = field(default_factory=dict, repr=False)

    @classmethod
    @sp.profile("byob.Background.build", tags=PROFILE_TAGS)
    def build(
        cls,
        grid: UniformGrid,
        probes: Sequence[Probe],
        walls: Sequence[Wall] = (),
        *,
        ion_mass: float = PROTON_MASS,
        presheath_debye_lengths: float = 20.0,
    ) -> Background:
        """Interpolate the ``probes`` over ``grid`` and set the bulk potential.

        Probes of ``density``, ``electron_temperature`` and ``ion_temperature``
        are required (one each is a uniform plasma). A ``potential`` probe fixes
        the plasma potential at its node; without one, it is zero where the
        density is largest. The presheath decays over ``presheath_debye_lengths``
        Debye lengths of the bulk.
        """
        points = grid.points().T
        solid = np.zeros(points.shape[0], dtype=bool)
        for wall in walls:
            solid |= wall.distance(points) < 0
        solid = solid.reshape(grid.num_gridpoints)

        bulk = {}
        for quantity in _BULK[:3]:
            selected = [p for p in probes if p.quantity == quantity]
            if not selected:
                raise ValueError(f"no {quantity!r} probe")
            bulk[quantity] = harmonic_fill(grid, selected, solid=solid)

        n, te = bulk["density"], bulk["electron_temperature"]
        anchors = [p for p in probes if p.quantity == "potential"]
        if anchors:
            node = grid.nearest_node(anchors[0].position)
            phi_ref = anchors[0].value
        else:
            node = np.unravel_index(np.nanargmax(n), n.shape)
            phi_ref = 0.0
        bulk["potential"] = phi_ref + te * np.log(n / n[node])
        return cls(grid, bulk, tuple(walls), ion_mass, presheath_debye_lengths)

    def _grid_values(self, quantity: str) -> np.ndarray:
        # sheaths only lower the densities, so the bulk density bounds them
        if quantity in ("electron_density", "ion_density"):
            quantity = "density"
        return self.bulk[quantity]

    def _interpolate(self, quantity: str, positions: np.ndarray) -> np.ndarray:
        if quantity not in self._interpolators:
            values = _fill_solid(self.bulk[quantity])
            self._interpolators[quantity] = RegularGridInterpolator(
                self.grid.axes, values, bounds_error=False, fill_value=None
            )
        return self._interpolators[quantity](positions)

    def _bulk_field(self, positions: np.ndarray) -> np.ndarray:
        key = "_grad"
        if key not in self._interpolators:
            values = _fill_solid(self.bulk["potential"])
            gradient = np.gradient(values, *self.grid.axes)
            gradient = [gradient] if self.grid.ndim == 1 else gradient
            self._interpolators[key] = [
                RegularGridInterpolator(
                    self.grid.axes, g, bounds_error=False, fill_value=None
                )
                for g in gradient
            ]
        return -np.stack([f(positions) for f in self._interpolators[key]], axis=-1)

    def _nearest_wall(self, positions: np.ndarray):
        """Return each position's nearest wall index and distance to it."""
        distances = np.stack([w.distance(positions) for w in self.walls])
        nearest = np.argmin(distances, axis=0)
        return nearest, distances[nearest, np.arange(positions.shape[0])]

    def _layers(self, positions: np.ndarray):
        """Yield ``(selection, wall, layer)`` for the wall nearest each position."""
        state = {q: self._interpolate(q, positions) for q in _BULK}
        nearest, distance = self._nearest_wall(positions)
        presheath = self.presheath_debye_lengths * debye_length(
            state["density"], state["electron_temperature"]
        )
        for index, wall in enumerate(self.walls):
            sel = nearest == index
            if not sel.any():
                continue
            bulk = PlasmaState(**{q: state[q][sel] for q in _BULK})
            yield (
                sel,
                wall,
                wall_layer(distance[sel], wall, bulk, self.ion_mass, presheath[sel]),
            )

    @sp.profile("byob.Background.evaluate", tags=PROFILE_TAGS)
    def evaluate(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        """Return densities, temperatures and potential at ``positions``.

        Keys: ``electron_density``, ``ion_density``, ``electron_temperature``,
        ``ion_temperature``, ``potential``, and ``density`` (the bulk density
        before the sheaths).
        """
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        out = {q: self._interpolate(q, positions) for q in _BULK}
        out["electron_density"] = out["density"].copy()
        out["ion_density"] = out["density"].copy()
        for sel, _, layer in self._layers(positions) if self.walls else ():
            for q in ("electron_density", "ion_density", "potential"):
                out[q][sel] = layer[q]
        return out

    @sp.profile("byob.Background.electric_field", tags=PROFILE_TAGS)
    def electric_field(self, positions: np.ndarray) -> np.ndarray:
        """Return the electric field ``(N, ndim)`` at ``positions``.

        The bulk part is ``-grad(phi)`` on the grid; the sheath part is
        ``-dphi/dd grad(d)``, with the gradient of the wall distance ``d`` from
        central differences.
        """
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        field_ = self._bulk_field(positions)
        if not self.walls:
            return field_
        step = 1e-3 * float(np.min(self.grid.spacing))
        for sel, wall, layer in self._layers(positions):
            p = positions[sel]
            normal = np.empty_like(p)
            for axis in range(p.shape[1]):
                shift = np.zeros(p.shape[1])
                shift[axis] = step
                normal[:, axis] = (
                    wall.distance(p + shift) - wall.distance(p - shift)
                ) / (2 * step)
            field_[sel] -= layer["dpotential"][:, None] * normal
        return field_

    def profile(self, quantity: str, reference: float | None = None) -> FieldProfile:
        """Return ``quantity`` over ``reference`` (the grid maximum by default).

        The result is a :class:`plasmacoll.background.Profile`, so it can shape
        a plasmacoll background.
        """
        if reference is None:
            reference = float(np.nanmax(self._grid_values(quantity)))
        return FieldProfile(self, quantity, reference)
