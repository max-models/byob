"""The live demo of the byob docs: the sheath and presheath in front of a wall.

It runs in the browser (Pyodide); `run` takes the values of the sliders and returns
the figures. A uniform plasma (one probe of each quantity) faces a wall at x = 0;
`Background.build` sets the bulk and `Background.evaluate` adds the Debye sheath
and presheath of the wall along a line normal to it.
"""

import numpy as np
from plasmacoll.constants import ATOMIC_MASS
from simgeometry import Rectangle

from byob import Background, Probe, UniformGrid, Wall, debye_length, sheath_width

ION_MASSES = {"H": 1.008, "He": 4.003, "Ar": 39.95}  # in u
WALLS = {"floating": None, "-30 V": -30.0, "-300 V": -300.0, "-3000 V": -3000.0}
ION_TEMPERATURE = 0.1  # eV
PRESHEATH_DEBYE_LENGTHS = 20.0  # byob's default


def _line(x, y, name, **style):
    return {"x": x, "y": y, "mode": "lines", "name": name, "line": style}


def run(density=1e17, electron_temperature=2.0, ion="H", wall="floating"):
    """Return the potential and densities in front of the wall.

    Args:
        density: Bulk plasma density (m^-3).
        electron_temperature: Bulk electron temperature (eV).
        ion: The ion species, a key of ``ION_MASSES``.
        wall: ``"floating"`` or the bias of the wall, a key of ``WALLS``.

    Returns:
        The demo's result: two Plotly figures (potential, densities), the
        Debye length, sheath width and drops, and a sentence on the sheath.
    """
    te = float(electron_temperature)
    debye = float(debye_length(density, te))
    # the layers are analytic, so a coarse grid of the uniform bulk will do
    extent = 500 * debye
    grid = UniformGrid((0.0, 0.0), (extent, extent / 4), (6, 3))
    shape = Rectangle((-extent, extent / 8), (2 * extent, 4 * extent))  # x < 0
    bias = WALLS[wall]
    walls = [Wall(shape) if bias is None else Wall(shape, "biased", bias)]
    centre = (extent / 2, extent / 8)
    probes = [
        Probe("density", centre, density),
        Probe("electron_temperature", centre, te),
        Probe("ion_temperature", centre, ION_TEMPERATURE),
    ]
    background = Background.build(
        grid,
        probes,
        walls,
        ion_mass=ION_MASSES[ion] * ATOMIC_MASS,
        presheath_debye_lengths=PRESHEATH_DEBYE_LENGTHS,
    )

    # the bulk potential is 0, the sheath edge at -Te/2
    wall_potential = float(background.evaluate([[0.0, extent / 8]])["potential"][0])
    drop = max(-0.5 * te - wall_potential, 0.0)
    width = float(sheath_width(drop, density * np.exp(-0.5), te))
    end = width + 3 * PRESHEATH_DEBYE_LENGTHS * debye
    x = np.linspace(0.0, end, 400)
    state = background.evaluate(np.stack([x, np.full_like(x, extent / 8)], axis=1))
    mm = 1e3 * x
    edge = {
        "type": "line",
        "x0": 1e3 * width,
        "x1": 1e3 * width,
        "yref": "paper",
        "y0": 0,
        "y1": 1,
        "line": {"dash": "dot", "width": 1},
    }
    xaxis = {"title": {"text": "distance from the wall (mm)"}, "range": [0, 1e3 * end]}

    potential = {
        "data": [_line(mm, state["potential"], "potential")],
        "layout": {
            "xaxis": xaxis,
            "yaxis": {"title": {"text": "potential (V)"}},
            "shapes": [edge],
            "showlegend": False,
        },
    }
    densities = {
        "data": [
            _line(mm, state["ion_density"] / density, "ions"),
            _line(mm, state["electron_density"] / density, "electrons", dash="dash"),
        ],
        "layout": {
            "xaxis": xaxis,
            "yaxis": {
                "title": {"text": "density / bulk density"},
                "type": "log",
                "range": [-3, 0.05],
            },
            "shapes": [edge],
        },
    }
    return {
        "figure": [potential, densities],
        "numbers": {
            "Debye length (µm)": 1e6 * debye,
            "Sheath width (µm)": 1e6 * width,
            "Sheath width / Debye length": width / debye,
            "Sheath drop (V)": drop,
            "Wall potential (V)": wall_potential,
        },
        "text": (
            f"The sheath drop is {drop / te:.1f} Te, the presheath adds 0.5 Te"
            if drop > 0
            else "The bias is above the sheath edge: no ion sheath, only the presheath"
        )
        + ".",
    }
