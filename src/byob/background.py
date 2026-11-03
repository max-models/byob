"""A static plasma background for particle tracking, built from probes and walls.

`Background.build` interpolates the probed density and temperatures over a
grid (`byob.probes.fill_probes`) and sets the bulk potential from the
Boltzmann relation ``phi = phi_ref + Te ln(n / n_ref)``. The bulk is gathered at
particle positions linearly by `mpifield.Field`'s compiled kernels; outside
the grid it takes the value of the nearest edge. Sheaths and presheaths
(`byob.sheath`) are added analytically at particle positions, so they stay
sharp however coarse the grid.

Positions are ``(N, ndim)`` in m, densities in m^-3, temperatures in eV,
potentials in V and electric fields in V/m.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scope_profiler as sp
from methodcard import ConvergenceProblem, Rule, method
from mpifield import Field
from plasmacoll.constants import PROTON_MASS
from scipy.ndimage import distance_transform_edt

from byob._profiling import PROFILE_TAGS
from byob.probes import Probe, UniformGrid, fill_probes
from byob.sheath import PlasmaState, Wall, debye_length, wall_layer

__all__ = ["Background", "FieldProfile"]

_UNITS = {
    "density": "m^-3",
    "electron_density": "m^-3",
    "ion_density": "m^-3",
    "electron_temperature": "eV",
    "ion_temperature": "eV",
    "potential": "V",
    "electric_field": "V/m",
}
_BULK = ("density", "electron_temperature", "ion_temperature", "potential")


def _fill_solid(values: np.ndarray) -> np.ndarray:
    """Return ``values`` with NaN nodes set to the nearest finite node."""
    missing = np.isnan(values)
    if not missing.any():
        return values
    index = distance_transform_edt(missing, return_distances=False, return_indices=True)
    return values[tuple(np.asarray(index))]


def _smooth_background(spacing: float) -> Background:
    """A wall-free background on the unit square with ``phi = sin(2x) cos(y)``."""
    count = round(1.0 / spacing) + 1
    grid = UniformGrid((0.0, 0.0), (1.0, 1.0), (count, count))
    x, y = np.meshgrid(*grid.axes, indexing="ij")
    potential = np.sin(2 * x) * np.cos(y)
    ones = np.ones_like(x)
    bulk = {
        "density": 1.0e17 * np.exp(potential),
        "electron_temperature": ones,
        "ion_temperature": ones,
        "potential": potential,
    }
    return Background(grid, bulk)


def _evaluate_problem() -> ConvergenceProblem:
    """The gathered potential of `_smooth_background` against the exact one."""
    points = np.random.default_rng(0).uniform(0.25, 0.75, size=(64, 2))
    exact = np.sin(2 * points[:, 0]) * np.cos(points[:, 1])

    def error(spacing: float) -> float:
        potential = _smooth_background(spacing).evaluate(points)["potential"]
        return float(np.max(np.abs(potential - exact)))

    return ConvergenceProblem(error, steps=[1 / 16, 1 / 32, 1 / 64, 1 / 128])


def _electric_field_problem() -> ConvergenceProblem:
    """The bulk field of `_smooth_background` against ``-grad(phi)``."""
    points = np.random.default_rng(0).uniform(0.25, 0.75, size=(64, 2))
    x, y = points.T
    exact = np.stack([-2 * np.cos(2 * x) * np.cos(y), np.sin(2 * x) * np.sin(y)], 1)

    def error(spacing: float) -> float:
        field_ = _smooth_background(spacing).electric_field(points)
        return float(np.max(np.abs(field_ - exact)))

    return ConvergenceProblem(error, steps=[1 / 16, 1 / 32, 1 / 64, 1 / 128])


def _build_rules() -> dict:
    """The SymPy rule of `Background.build`: a two-node grid with a potential probe."""
    import sympy

    n, n_ref, te, phi_ref = sympy.symbols(
        "density reference_density te reference_potential", positive=True
    )

    def boltzmann(density, reference_density, te, reference_potential):
        grid = UniformGrid((0.0,), (1.0,), (2,))
        probes = [
            Probe("density", (0.0,), density),
            Probe("density", (1.0,), reference_density),
            Probe("electron_temperature", (0.0,), te),
            Probe("ion_temperature", (0.0,), te),
            Probe("potential", (1.0,), reference_potential),
        ]
        return Background.build(grid, probes).bulk["potential"][0]

    ranges = {
        "density": (1.0e15, 1.0e19),
        "reference_density": (1.0e15, 1.0e19),
        "te": (0.5, 20.0),
        "reference_potential": (-10.0, 10.0),
    }
    rule = phi_ref + te * sympy.log(n / n_ref)
    return {"boltzmann": Rule(rule, boltzmann, ranges, rtol=1e-9)}


@dataclass(frozen=True)
@method(
    equations={
        "factor": r"f(\mathbf x) = \frac{Q(\mathbf x)}{Q_\mathrm{ref}}, \qquad f_\mathrm{max} = \frac{\max_k Q_k}{Q_\mathrm{ref}}",
    },
    assumptions=(
        r"$Q_\mathrm{ref}$ the grid maximum of the bulk value by default (`Background.profile`)",
        r"$n_e$ and $n_i$ bounded by the bulk density on the grid: the layers only lower them",
    ),
)
class FieldProfile:
    """A background quantity over a reference value: a `plasmacoll` Profile.

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
@method(
    equations={
        "potential": r"\phi(\mathbf x) = \phi_\mathrm{bulk}(\mathbf x) + f_{w^*}\big(d_{w^*}(\mathbf x)\big), \qquad f_w(d) = \phi_w(d) - \phi_\mathrm{bulk}",
        "state": r"\big(n, T_e, T_i, \phi_\mathrm{bulk}\big)(\mathbf x) \text{ gathered from the grid}, \qquad (n_e, n_i, \phi) \text{ from the layer of the controlling wall } w^*",
    },
    assumptions=(
        r"static: nothing is solved self-consistently, no Poisson equation, no transport, no magnetic field",
        r"the bulk interpolated from probes on a uniform grid, its potential from Boltzmann's relation; the Debye sheath and presheath of every wall added analytically at the evaluation point",
        r"positions in m, densities in m$^{-3}$, temperatures in eV, potentials in V, fields in V/m",
    ),
    components=(
        "byob.background.Background.build",
        "byob.background.Background.evaluate",
        "byob.background.Background.electric_field",
    ),
)
class Background:
    """Bulk fields on a grid, with sheaths at ``walls`` added at evaluation."""

    grid: UniformGrid
    bulk: dict[str, np.ndarray]
    walls: Sequence[Wall] = ()
    ion_mass: float = PROTON_MASS
    presheath_debye_lengths: float = 20.0
    probes: Sequence[Probe] = ()
    _fields: dict[str, Field] = field(default_factory=dict, repr=False)

    @classmethod
    @method(
        equations={
            "solid": r"\text{node } k \text{ solid if } d_w(\mathbf x_k) < 0 \text{ for a wall } w",
            "boltzmann": r"\phi_k = \phi_\mathrm{ref} + T_{e,k} \ln\frac{n_k}{n_\mathrm{ref}} \quad\Longleftrightarrow\quad n = n_\mathrm{ref} \exp\frac{\phi - \phi_\mathrm{ref}}{T_e}",
        },
        assumptions=(
            r"$n$, $T_e$ and $T_i$ filled from their probes, solid nodes left out (`fill_probes`)",
            r"$n_\mathrm{ref}$ at the node nearest the first `potential` probe and $\phi_\mathrm{ref}$ its value (further potential probes ignored); without one, the node of largest density and $\phi_\mathrm{ref} = 0$",
            r"the local $T_{e,k}$ multiplies the logarithm: exact for a uniform $T_e$ only",
            r"a `potential` probe on a wall node makes $\phi$ NaN everywhere",
        ),
        references=(
            "[Lieberman and Lichtenberg, Principles of Plasma Discharges and Materials Processing, 2nd ed. (Wiley, 2005)](https://doi.org/10.1002/0471724254), ch. 6",
        ),
        components=(fill_probes,),
        symbolic=_build_rules,
    )
    @sp.profile("byob.Background.build", tags=PROFILE_TAGS)
    def build(
        cls,
        grid: UniformGrid,
        probes: Sequence[Probe],
        walls: Sequence[Wall] = (),
        *,
        ion_mass: float = PROTON_MASS,
        presheath_debye_lengths: float = 20.0,
        method: Literal["biharmonic", "harmonic"] = "biharmonic",
    ) -> Background:
        """Interpolate the ``probes`` over ``grid`` and set the bulk potential.

        Probes of ``density``, ``electron_temperature`` and ``ion_temperature``
        are required (one each is a uniform plasma). A ``potential`` probe fixes
        the plasma potential at its node; without one, it is zero where the
        density is largest. The presheath decays over ``presheath_debye_lengths``
        Debye lengths of the bulk. ``method`` picks the interpolation, see
        `byob.probes.fill_probes`.
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
            bulk[quantity] = fill_probes(grid, selected, solid=solid, method=method)

        n, te = bulk["density"], bulk["electron_temperature"]
        anchors = [p for p in probes if p.quantity == "potential"]
        if anchors:
            node = grid.nearest_node(anchors[0].position)
            phi_ref = anchors[0].value
        else:
            node = np.unravel_index(np.nanargmax(n), n.shape)
            phi_ref = 0.0
        bulk["potential"] = phi_ref + te * np.log(n / n[node])
        return cls(
            grid, bulk, tuple(walls), ion_mass, presheath_debye_lengths, tuple(probes)
        )

    def _grid_values(self, quantity: str) -> np.ndarray:
        # sheaths only lower the densities, so the bulk density bounds them
        if quantity in ("electron_density", "ion_density"):
            quantity = "density"
        return self.bulk[quantity]

    def _field(self, data: np.ndarray) -> Field:
        return Field.from_array(data, self.grid.lower_bound, self.grid.upper_bound)

    def _bulk_state(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        """Return the bulk quantities at ``positions``, gathered in one pass."""
        if "bulk" not in self._fields:
            stacked = np.stack([_fill_solid(self.bulk[q]) for q in _BULK], axis=-1)
            self._fields["bulk"] = self._field(stacked)
        values = np.asarray(self._fields["bulk"].gather(positions.T))
        return dict(zip(_BULK, values, strict=True))

    def _bulk_field(self, positions: np.ndarray) -> np.ndarray:
        if "gradient" not in self._fields:
            potential = self._field(_fill_solid(self.bulk["potential"]))
            self._fields["gradient"] = potential.gradient()
        return -np.asarray(self._fields["gradient"].gather(positions.T)).T

    def _nearest_wall(self, positions: np.ndarray):
        """Return each position's nearest wall index and distance to it."""
        distances = np.stack([w.distance(positions) for w in self.walls])
        nearest = np.argmin(distances, axis=0)
        return nearest, distances[nearest, np.arange(positions.shape[0])]

    def _layers(self, positions: np.ndarray):
        """Yield ``(selection, wall, layer)``, the controlling wall at each position.

        Every wall's layer is evaluated everywhere and each position takes the
        one with the lowest potential. That is the nearest wall away from
        corners, and, unlike the nearest wall, it keeps the potential
        continuous where two presheaths overlap.
        """
        state = self._bulk_state(positions)
        bulk = PlasmaState(**state)
        presheath = self.presheath_debye_lengths * debye_length(
            state["density"], state["electron_temperature"]
        )
        layers = [
            wall_layer(w.distance(positions), w, bulk, self.ion_mass, presheath)
            for w in self.walls
        ]
        controlling = np.argmin([layer["potential"] for layer in layers], axis=0)
        for index, (wall, layer) in enumerate(zip(self.walls, layers, strict=True)):
            sel = controlling == index
            if sel.any():
                yield sel, wall, {q: v[sel] for q, v in layer.items()}

    @method(
        equations={
            "gather": r"Q(\mathbf x) = \sum_k Q_k \prod_d \max\left(0,\, 1 - \frac{|x_d - x_{d,k}|}{h_d}\right)",
            "presheath_length": r"L = N_\lambda\, \lambda_D(n, T_e)",
            "controlling_wall": r"w^*(\mathbf x) = \arg\min_w \phi_w\big(d_w(\mathbf x)\big)",
            "layer": r"(n_e, n_i, \phi)(\mathbf x) = \text{the layer of } w^* \text{ at } d_{w^*}(\mathbf x), \qquad n_e = n_i = n \text{ without walls}",
        },
        order=2,
        assumptions=(
            r"multilinear gather by mpifield's compiled kernels; outside the grid the edge value",
            r"wall and unreached nodes take the nearest finite node first (Euclidean distance transform in index units)",
            r"the controlling wall, of lowest potential, is the nearest one away from corners and keeps $\phi$ continuous where presheaths overlap",
            r"`density` keeps the bulk value; the temperatures are unchanged by the layers",
            r"the order is that of the gather of a smooth bulk inside the grid",
        ),
        components=(wall_layer, debye_length),
        problem=_evaluate_problem,
    )
    @sp.profile("byob.Background.evaluate", tags=PROFILE_TAGS)
    def evaluate(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        """Return densities, temperatures and potential at ``positions``.

        Keys: ``electron_density``, ``ion_density``, ``electron_temperature``,
        ``ion_temperature``, ``potential``, and ``density`` (the bulk density
        before the sheaths).
        """
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        out = self._bulk_state(positions)
        out["electron_density"] = out["density"].copy()
        out["ion_density"] = out["density"].copy()
        for sel, _, layer in self._layers(positions) if self.walls else ():
            for q in ("electron_density", "ion_density", "potential"):
                out[q][sel] = layer[q]
        return out

    @method(
        equations={
            "field": r"\mathbf E = -\nabla \phi_\mathrm{bulk} - \frac{d\phi}{dd}\, \nabla d",
            "bulk_gradient": r"(\partial_d \phi)_k = \frac{\phi_{k + \mathbf e_d} - \phi_{k - \mathbf e_d}}{2 h_d}",
            "normal": r"\partial_j d = \frac{d(\mathbf x + \epsilon\, \mathbf e_j) - d(\mathbf x - \epsilon\, \mathbf e_j)}{2 \epsilon}, \qquad \epsilon = 10^{-3} \min_j h_j",
        },
        order=2,
        assumptions=(
            r"the chain rule for $\phi = \phi_\mathrm{bulk} + f(d)$ with $n$, $T_e$, $\Delta\phi$ and $s$ held fixed: the bulk gradients inside the layer are left out",
            r"$d\phi/dd$ of the controlling wall (`wall_layer`); zero without walls, discontinuous at the sheath edge",
            r"the bulk gradient by `mpifield.Field.gradient`, one-sided (first order) at the grid ends and using the filled values inside walls; gathered multilinearly",
            r"the order is that of the bulk field of a smooth potential inside the grid",
        ),
        components=(wall_layer,),
        problem=_electric_field_problem,
    )
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

    @sp.profile("byob.Background.sample", tags=PROFILE_TAGS)
    def sample(self, x=None, y=None, z=None, *, resolution=128):
        """Return every quantity on a tensor grid as an `xarray.Dataset`.

        Like `simgeometry.api.GeometryAPI.evaluate`: a scalar fixes an
        axis, a sequence samples it, and an axis left out gets ``resolution``
        points across the grid bounds. Fixed axes are dropped from the
        dimensions, so ``sample(y=0.0)`` is a line and ``sample()`` the whole
        domain. The values come from `evaluate` and
        `electric_field`, with the field's
        ``component`` dimension as plasma-plots' ``vector`` expects.
        """
        import xarray as xr

        names = "xyz"[: self.grid.ndim]
        given = dict(zip("xyz", (x, y, z), strict=True))
        coords = {}
        for name, lo, hi in zip(
            names, self.grid.lower_bound, self.grid.upper_bound, strict=True
        ):
            value = given[name]
            coords[name] = (
                np.linspace(lo, hi, resolution)
                if value is None
                else np.asarray(value, dtype=float)
            )
        mesh = np.meshgrid(*(np.atleast_1d(c) for c in coords.values()), indexing="ij")
        positions = np.stack([m.ravel() for m in mesh], axis=-1)
        dims = [n for n in names if coords[n].ndim == 1]
        shape = [coords[n].size for n in dims]

        values = self.evaluate(positions)
        field_ = self.electric_field(positions)
        if self.walls:
            _, distance = self._nearest_wall(positions)
            for v in values.values():
                v[distance < 0] = np.nan  # inside a wall
            field_[distance < 0] = np.nan
        data = {q: (dims, v.reshape(shape)) for q, v in values.items()}
        data["electric_field"] = (["component", *dims], field_.T.reshape(-1, *shape))
        dataset = xr.Dataset(
            data, coords={n: coords[n] for n in names} | {"component": list(names)}
        )
        for name, unit in _UNITS.items():
            dataset[name].attrs["units"] = unit
        return dataset

    def profile(self, quantity: str, reference: float | None = None) -> FieldProfile:
        """Return ``quantity`` over ``reference`` (the grid maximum by default).

        The result is a `plasmacoll.background.Profile`, so it can shape
        a plasmacoll background.
        """
        if reference is None:
            reference = float(np.nanmax(self._grid_values(quantity)))
        return FieldProfile(self, quantity, reference)
