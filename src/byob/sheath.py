"""Debye sheaths and presheaths in front of walls, from the local plasma state.

Temperatures are in eV and potentials in V, so ``T/e`` is just ``T``. The
sheath is not resolved on the grid: its potential drop, width and profile follow
from the electron temperature and density next to the wall, and are evaluated
analytically at the distance of a point to the wall.

Going from the bulk to the wall, the potential first falls by ``Te/2`` across
the presheath, where ions are accelerated to the Bohm speed, then by
:func:`sheath_drop` across a Child-Langmuir sheath of width :func:`sheath_width`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

__all__ = [
    "PlasmaState",
    "Wall",
    "debye_length",
    "sheath_drop",
    "sheath_width",
    "wall_layer",
]

VACUUM_PERMITTIVITY = 8.8541878128e-12


def debye_length(density, electron_temperature):
    """Return the electron Debye length (m) for ``density`` (m^-3) and ``Te`` (eV)."""
    return np.sqrt(
        VACUUM_PERMITTIVITY
        * np.asarray(electron_temperature)
        / (ELEMENTARY_CHARGE * np.asarray(density))
    )


@dataclass(frozen=True)
class Wall:
    """A wall: a geometry with ``signed_distance`` (negative inside) and its sheath.

    A ``"floating"`` wall draws no net current. A ``"biased"`` wall sits at
    ``bias`` (V); if that is above the sheath edge potential, it gets no sheath,
    since electron sheaths are not modelled.
    """

    shape: Any
    kind: Literal["floating", "biased"] = "floating"
    bias: float = 0.0

    def distance(self, positions: np.ndarray) -> np.ndarray:
        """Return the signed distance of the ``(N, ndim)`` positions to the wall."""
        return np.asarray(self.shape.signed_distance(np.asarray(positions).T))


@dataclass(frozen=True)
class PlasmaState:
    """Bulk plasma at some points: densities in m^-3, temperatures in eV, V."""

    density: np.ndarray
    electron_temperature: np.ndarray
    ion_temperature: np.ndarray
    potential: np.ndarray


def sheath_drop(wall: Wall, bulk: PlasmaState, ion_mass: float) -> np.ndarray:
    """Return the potential drop (V) across the sheath, from its edge to the wall."""
    te = np.asarray(bulk.electron_temperature)
    if wall.kind == "floating":
        ratio = ion_mass / (
            2 * math.pi * ELECTRON_MASS * (1 + bulk.ion_temperature / te)
        )
        return 0.5 * te * np.log(ratio)
    edge = np.asarray(bulk.potential) - 0.5 * te
    return np.maximum(edge - wall.bias, 0.0)


def sheath_width(drop, edge_density, electron_temperature):
    """Return the Child-Langmuir sheath width (m) for a ``drop`` (V)."""
    te = np.asarray(electron_temperature)
    return (
        math.sqrt(2)
        / 3
        * debye_length(edge_density, te)
        * (2 * np.asarray(drop) / te) ** 0.75
    )


def wall_layer(
    distance: np.ndarray,
    wall: Wall,
    bulk: PlasmaState,
    ion_mass: float,
    presheath_length: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Return the presheath and sheath at ``distance`` (m) from ``wall``.

    The bulk state is that of the points themselves, taken as constant across
    the layer. Returns the electron and ion densities, the potential and its
    derivative ``dpotential`` with respect to the distance. Points in the
    presheath fall off towards the bulk over ``presheath_length``.
    """
    te = np.asarray(bulk.electron_temperature, dtype=float)
    n = np.asarray(bulk.density, dtype=float)
    d = np.maximum(np.asarray(distance, dtype=float), 0.0)
    drop = sheath_drop(wall, bulk, ion_mass)
    edge_density = n * math.exp(-0.5)
    width = sheath_width(drop, edge_density, te)
    edge_potential = np.asarray(bulk.potential) - 0.5 * te

    # presheath: phi = phi_bulk - Te/2 exp(-(d - s)/L), Boltzmann electrons
    decay = np.exp(-np.maximum(d - width, 0.0) / presheath_length)
    potential = bulk.potential - 0.5 * te * decay
    dpotential = 0.5 * te * decay / presheath_length
    electrons = n * np.exp(-0.5 * decay)
    ions = electrons.copy()

    # sheath: phi = phi_wall + drop (d/s)^(4/3), cold ions with the Bohm flux
    inside = d < width
    if np.any(inside):
        s, dr = width[inside], drop[inside]
        x = d[inside] / s
        below_edge = dr * (1 - x ** (4 / 3))
        potential[inside] = edge_potential[inside] - below_edge
        dpotential[inside] = dr * (4 / 3) * x ** (1 / 3) / s
        ne, t = edge_density[inside], te[inside]
        electrons[inside] = ne * np.exp(-below_edge / t)
        ions[inside] = ne / np.sqrt(1 + 2 * below_edge / t)
    return {
        "electron_density": electrons,
        "ion_density": ions,
        "potential": potential,
        "dpotential": dpotential,
    }
