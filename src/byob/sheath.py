"""Debye sheaths and presheaths in front of walls, from the local plasma state.

Temperatures are in eV and potentials in V, so ``T/e`` is just ``T``. The
sheath is not resolved on the grid: its potential drop, width and profile follow
from the bulk electron temperature and density at a point, taken as constant
across the layer, and are evaluated analytically at the distance of that point
to the wall.

Going from the bulk to the wall, the potential first falls by ``Te/2`` across
the presheath, where ions are accelerated to the Bohm speed, then by
`sheath_drop` across a Child-Langmuir sheath of width `sheath_width`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from methodcard import Rule, method
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


def _debye_length_rules() -> dict:
    """The SymPy rule of `debye_length` (for the tests)."""
    import sympy

    n, te = sympy.symbols("density electron_temperature", positive=True)
    eps0, e = sympy.Float(VACUUM_PERMITTIVITY), sympy.Float(ELEMENTARY_CHARGE)
    ranges = {"density": (1.0e14, 1.0e20), "electron_temperature": (0.1, 100.0)}
    return {"debye": Rule(sympy.sqrt(eps0 * te / (e * n)), ranges=ranges)}


@method(
    equations={
        "debye": r"\lambda_D = \sqrt{\frac{\varepsilon_0 T_e}{e\, n}}",
    },
    assumptions=(
        r"$T_e$ in eV, so $k T_e / e^2$ is $T_e / e$",
        r"$\varepsilon_0 = 8.8541878128 \times 10^{-12}$ F/m",
    ),
    references=(
        "[Lieberman and Lichtenberg, Principles of Plasma Discharges and Materials Processing, 2nd ed. (Wiley, 2005)](https://doi.org/10.1002/0471724254), ch. 6",
    ),
    symbolic=_debye_length_rules,
)
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


def _sheath_drop_rules() -> dict:
    """The SymPy rules of `sheath_drop` (for the tests)."""
    import sympy

    te, ti, mi = sympy.symbols("te ti ion_mass", positive=True)
    phi, bias = sympy.symbols("phi bias", real=True)
    me = sympy.Float(ELECTRON_MASS)

    def floating(te, ti, ion_mass):
        bulk = PlasmaState(np.ones(1), np.full(1, te), np.full(1, ti), np.zeros(1))
        return sheath_drop(Wall(None), bulk, ion_mass)[0]

    def biased(phi, te, bias):
        bulk = PlasmaState(np.ones(1), np.full(1, te), np.zeros(1), np.full(1, phi))
        return sheath_drop(Wall(None, "biased", bias), bulk, 1.0)[0]

    return {
        "floating": Rule(
            te / 2 * sympy.log(mi / (2 * sympy.pi * me * (1 + ti / te))),
            implementation=floating,
            ranges={
                "te": (0.5, 20.0),
                "ti": (0.0, 20.0),
                "ion_mass": (1.6e-27, 6.7e-26),
            },
        ),
        "biased": Rule(
            sympy.Max(phi - te / 2 - bias, 0),
            implementation=biased,
            ranges={"phi": (-10.0, 10.0), "te": (0.5, 20.0), "bias": (-40.0, 20.0)},
        ),
    }


@method(
    equations={
        "fluxes": r"n_s u_B = \frac14 n_s \bar v_e\, e^{-\Delta\phi / T_e}, \qquad u_B = \sqrt{\frac{T_e + T_i}{m_i}}, \qquad \bar v_e = \sqrt{\frac{8 T_e}{\pi m_e}}",
        "floating": r"\Delta\phi_\mathrm{floating} = \frac{T_e}{2} \ln \frac{m_i}{2\pi m_e (1 + T_i/T_e)}",
        "biased": r"\Delta\phi_\mathrm{biased} = \max(\phi_s - V_\mathrm{bias},\, 0), \qquad \phi_s = \phi_\mathrm{bulk} - \frac{T_e}{2}",
    },
    assumptions=(
        r"a floating wall draws no net current: Maxwellian electrons, one singly charged ion species, no secondary electron emission",
        r"$T_i$ enters only here, through the Bohm speed of the ion flux; the presheath drop and the sheath densities keep the cold-ion Bohm speed",
        r"no electron sheaths: a wall biased above the sheath edge, $V_\mathrm{bias} > \phi_s$, gets $\Delta\phi = 0$",
        r"for hydrogen and $T_i = 0$, $\Delta\phi_\mathrm{floating} \approx 2.84\, T_e$, so $\phi_\mathrm{wall} \approx \phi_\mathrm{bulk} - 3.3\, T_e$",
    ),
    references=(
        "[Stangeby, The Plasma Boundary of Magnetic Fusion Devices (IoP, 2000)](https://doi.org/10.1201/9781420033328), ch. 2",
        "D. Bohm, in Guthrie and Wakerling (eds.), The Characteristics of Electrical Discharges in Magnetic Fields (McGraw-Hill, 1949), ch. 3",
    ),
    symbolic=_sheath_drop_rules,
)
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


def _sheath_width_rules() -> dict:
    """The SymPy rule of `sheath_width` (for the tests)."""
    import sympy

    drop, ns, te = sympy.symbols(
        "drop edge_density electron_temperature", positive=True
    )
    eps0, e = sympy.Float(VACUUM_PERMITTIVITY), sympy.Float(ELEMENTARY_CHARGE)
    debye = sympy.sqrt(eps0 * te / (e * ns))
    rule = sympy.sqrt(2) / 3 * debye * (2 * drop / te) ** sympy.Rational(3, 4)
    ranges = {
        "drop": (0.0, 200.0),
        "edge_density": (1.0e14, 1.0e20),
        "electron_temperature": (0.5, 20.0),
    }
    return {"width": Rule(rule, ranges=ranges)}


@method(
    equations={
        "child_langmuir": r"\frac{4}{9} \varepsilon_0 \sqrt{\frac{2e}{m_i}}\, \frac{\Delta\phi^{3/2}}{s^2} = e\, n_s u_B, \qquad u_B = \sqrt{\frac{T_e}{m_i}}",
        "width": r"s = \frac{\sqrt 2}{3}\, \lambda_D(n_s, T_e) \left(\frac{2\Delta\phi}{T_e}\right)^{3/4}",
    },
    assumptions=(
        r"cold ions entering at the Bohm speed; the space charge of the ions only",
        r"the Debye length of the edge density $n_s$",
        r"Child-Langmuir holds for $\Delta\phi \gg T_e$; applied to any $\Delta\phi \ge 0$, so the width goes to 0 as $\Delta\phi^{3/4}$",
    ),
    references=(
        "[Lieberman and Lichtenberg, Principles of Plasma Discharges and Materials Processing, 2nd ed. (Wiley, 2005)](https://doi.org/10.1002/0471724254), ch. 6",
        "[C. D. Child, Phys. Rev. (Series I) 32, 492 (1911)](https://doi.org/10.1103/PhysRevSeriesI.32.492)",
        "[I. Langmuir, Phys. Rev. 2, 450 (1913)](https://doi.org/10.1103/PhysRev.2.450)",
    ),
    components=(debye_length,),
    symbolic=_sheath_width_rules,
)
def sheath_width(drop, edge_density, electron_temperature):
    """Return the Child-Langmuir sheath width (m) for a ``drop`` (V)."""
    te = np.asarray(electron_temperature)
    return (
        math.sqrt(2)
        / 3
        * debye_length(edge_density, te)
        * (2 * np.asarray(drop) / te) ** 0.75
    )


def _wall_layer_rules() -> dict:
    """The SymPy rules of `wall_layer`, on a biased wall with a given drop (tests)."""
    import sympy

    n, te, drop, xi, offset, length = sympy.symbols(
        "density te drop xi offset length", positive=True
    )
    phi = sympy.Symbol("phi", real=True)
    eps0, e = sympy.Float(VACUUM_PERMITTIVITY), sympy.Float(ELEMENTARY_CHARGE)
    ns = n * sympy.exp(-sympy.Rational(1, 2))
    width = (
        sympy.sqrt(2)
        / 3
        * sympy.sqrt(eps0 * te / (e * ns))
        * (2 * drop / te) ** sympy.Rational(3, 4)
    )
    below_edge = drop * (1 - xi ** sympy.Rational(4, 3))
    decay = sympy.exp(-offset / length)

    def layer(quantity, density, te, phi, drop, distance, length=1.0):
        bulk = PlasmaState(
            np.full(1, density), np.full(1, te), np.zeros(1), np.full(1, phi)
        )
        wall = Wall(None, "biased", phi - 0.5 * te - drop)
        result = wall_layer(np.full(1, distance), wall, bulk, 1.0, length)
        return result[quantity][0]

    def in_sheath(quantity):
        def call(density=1.0e17, te=2.0, phi=0.0, drop=10.0, xi=0.5):
            width = sheath_width(drop, density * math.exp(-0.5), te)
            return layer(quantity, density, te, phi, drop, xi * width)

        return call

    def in_presheath(quantity):
        def call(
            density=1.0e17, te=2.0, phi=0.0, drop=10.0, offset=1.0e-3, length=1.0e-3
        ):
            width = sheath_width(drop, density * math.exp(-0.5), te)
            return layer(quantity, density, te, phi, drop, width + offset, length)

        return call

    sheath = {
        "density": (1.0e15, 1.0e19),
        "te": (0.5, 20.0),
        "phi": (-10.0, 10.0),
        "drop": (1.0, 100.0),
        "xi": (0.01, 0.99),
    }
    presheath = sheath | {"offset": (0.0, 5.0e-3), "length": (1.0e-4, 1.0e-2)}
    del presheath["xi"]
    return {
        "sheath": Rule(
            phi - te / 2 - below_edge, in_sheath("potential"), sheath, rtol=1e-9
        ),
        "sheath_field": Rule(
            sympy.Rational(4, 3) * drop / width * xi ** sympy.Rational(1, 3),
            in_sheath("dpotential"),
            sheath,
            rtol=1e-9,
        ),
        "sheath_electrons": Rule(
            ns * sympy.exp(-below_edge / te),
            in_sheath("electron_density"),
            sheath,
            rtol=1e-9,
        ),
        "sheath_ions": Rule(
            ns / sympy.sqrt(1 + 2 * below_edge / te),
            in_sheath("ion_density"),
            sheath,
            rtol=1e-9,
        ),
        "presheath": Rule(
            phi - te / 2 * decay, in_presheath("potential"), presheath, rtol=1e-9
        ),
        "presheath_field": Rule(
            te / (2 * length) * decay,
            in_presheath("dpotential"),
            presheath,
            rtol=1e-9,
        ),
        "presheath_density": Rule(
            n * sympy.exp(-decay / 2),
            in_presheath("ion_density"),
            presheath,
            rtol=1e-9,
        ),
    }


@method(
    equations={
        "edge": r"\phi_s = \phi_\mathrm{bulk} - \frac{T_e}{2}, \qquad n_s = n\, e^{-1/2}, \qquad \tfrac12 m_i u_B^2 = e\, \frac{T_e}{2}",
        "sheath": r"\phi = \phi_s - \Delta\phi\,(1 - \xi^{4/3}) = \phi_\mathrm{wall} + \Delta\phi\, \xi^{4/3}, \qquad \xi = \frac{d}{s}, \quad 0 \le d < s",
        "sheath_field": r"\frac{d\phi}{dd} = \frac{4}{3} \frac{\Delta\phi}{s}\, \xi^{1/3}",
        "sheath_electrons": r"n_e = n_s \exp\left(-\frac{\phi_s - \phi}{T_e}\right)",
        "sheath_ions": r"n_i = \frac{n_s}{\sqrt{1 + 2(\phi_s - \phi)/T_e}}, \qquad n_i u_i = n_s u_B",
        "presheath": r"\phi = \phi_\mathrm{bulk} - \frac{T_e}{2} \exp\left(-\frac{d - s}{L}\right), \qquad d \ge s",
        "presheath_field": r"\frac{d\phi}{dd} = \frac{T_e}{2L} \exp\left(-\frac{d - s}{L}\right)",
        "presheath_density": r"n_e = n_i = n \exp\left(\frac{\phi - \phi_\mathrm{bulk}}{T_e}\right)",
    },
    assumptions=(
        r"the bulk state ($n$, $T_e$, $\phi_\mathrm{bulk}$) of the point itself, taken as constant across the layer: $\lambda_D \lesssim s \ll L \ll$ the bulk gradient length (not checked)",
        r"cold ions reach the Bohm speed $u_B = \sqrt{T_e/m_i}$ at the sheath edge; Boltzmann electrons; temperatures unchanged across the layer",
        r"the $4/3$ power is taken about the wall, not about the sheath edge as in the Child-Langmuir solution: zero field at the wall, $\tfrac43 \Delta\phi/s$ at the edge, where $d\phi/dd$ jumps to $T_e/(2L)$",
        r"no Poisson solve: the Child-Langmuir potential and the Boltzmann and Bohm densities are not mutually consistent",
        r"$d < 0$ (inside the wall) is clipped to 0, the wall values; with $\Delta\phi = 0$ the presheath only",
        r"$L = N_\lambda \lambda_D(n, T_e)$ is given by the caller (`presheath_debye_lengths` of `Background`, 20 by default)",
    ),
    references=(
        "[Stangeby, The Plasma Boundary of Magnetic Fusion Devices (IoP, 2000)](https://doi.org/10.1201/9781420033328), ch. 2",
        "[Lieberman and Lichtenberg, Principles of Plasma Discharges and Materials Processing, 2nd ed. (Wiley, 2005)](https://doi.org/10.1002/0471724254), ch. 6",
    ),
    components=(sheath_drop, sheath_width),
    symbolic=_wall_layer_rules,
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
