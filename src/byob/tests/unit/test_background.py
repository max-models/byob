import math

import numpy as np
import pytest
from plasmacoll.constants import ELECTRON_MASS, PROTON_MASS

from byob import Background, Probe, UniformGrid, Wall, harmonic_fill, sheath_drop
from byob.sheath import PlasmaState


class HalfSpace:
    """Solid for x < x0; the signed distance is exact."""

    def __init__(self, x0):
        self.x0 = x0

    def signed_distance(self, points):
        return np.asarray(points)[0] - self.x0


GRID = UniformGrid((0.0, 0.0), (0.1, 0.05), (41, 21))


def uniform_probes(n=1e17, te=2.0, ti=0.5):
    centre = (0.05, 0.025)
    return [
        Probe("density", centre, n),
        Probe("electron_temperature", centre, te),
        Probe("ion_temperature", centre, ti),
    ]


def test_harmonic_fill_hits_probes_without_overshoot():
    probes = [Probe("density", (0.0, 0.0), 1e16), Probe("density", (0.1, 0.05), 1e18)]
    values = harmonic_fill(GRID, probes)
    assert values[0, 0] == pytest.approx(1e16)
    assert values[-1, -1] == pytest.approx(1e18)
    assert values.min() >= 1e16 * (1 - 1e-12)
    assert values.max() <= 1e18 * (1 + 1e-12)


def test_harmonic_fill_unreached_region_is_nan():
    solid = np.zeros(GRID.num_gridpoints, dtype=bool)
    solid[20, :] = True  # a wall splitting the domain
    values = harmonic_fill(GRID, [Probe("density", (0.01, 0.01), 1e17)], solid=solid)
    assert np.all(np.isnan(values[21:]))
    assert np.allclose(values[:20], 1e17)


def test_floating_sheath_drop_hydrogen():
    bulk = PlasmaState(
        np.array([1e17]), np.array([2.0]), np.array([0.0]), np.array([0.0])
    )
    drop = sheath_drop(Wall(HalfSpace(0.0)), bulk, PROTON_MASS)
    expected = 0.5 * 2.0 * math.log(PROTON_MASS / (2 * math.pi * ELECTRON_MASS))
    assert drop[0] == pytest.approx(expected)  # about 2.8 Te


def test_floating_wall_potential_and_field():
    te = 2.0
    wall = Wall(HalfSpace(0.01))
    bg = Background.build(GRID, uniform_probes(te=te, ti=0.5), [wall])
    bulk = bg.evaluate([[0.08, 0.025]])
    assert bulk["potential"][0] == pytest.approx(0.0, abs=1e-6)
    at_wall = bg.evaluate([[0.01, 0.025]])
    drop = (
        0.5
        * te
        * math.log(PROTON_MASS / (2 * math.pi * ELECTRON_MASS * (1 + 0.5 / te)))
    )
    assert at_wall["potential"][0] == pytest.approx(-0.5 * te - drop)
    assert at_wall["ion_density"][0] > at_wall["electron_density"][0]

    # field points into the wall (-x) near it and is consistent with -dphi/dx
    x = 0.01 + 1e-6
    e = bg.electric_field([[x, 0.025]])[0]
    h = 1e-9
    phi = bg.evaluate([[x - h, 0.025], [x + h, 0.025]])["potential"]
    assert e[0] < 0
    assert e[0] == pytest.approx(-(phi[1] - phi[0]) / (2 * h), rel=1e-3)
    assert e[1] == pytest.approx(0.0, abs=1e-6)


def test_biased_wall_above_plasma_has_no_sheath():
    wall = Wall(HalfSpace(0.01), kind="biased", bias=100.0)
    bg = Background.build(GRID, uniform_probes(), [wall])
    at_wall = bg.evaluate([[0.01, 0.025]])
    assert at_wall["potential"][0] == pytest.approx(-1.0)  # presheath only


def test_profile_is_plasmacoll_profile():
    from plasmacoll.background import Profile

    bg = Background.build(GRID, uniform_probes(), [Wall(HalfSpace(0.01))])
    profile = bg.profile("electron_density")
    assert isinstance(profile, Profile)
    assert profile.max_factor == pytest.approx(1.0)
    assert np.all(profile(np.array([[0.05, 0.02], [0.0101, 0.02]])) <= 1.0)
