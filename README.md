# byob


<!-- README.md is generated from README.qmd: edit the .qmd and run `make readme`. -->

[![Tests](https://github.com/max-models/byob/actions/workflows/test_pytest.yml/badge.svg)](https://github.com/max-models/byob/actions/workflows/test_pytest.yml)
[![Static
analysis](https://github.com/max-models/byob/actions/workflows/static_analysis.yml/badge.svg)](https://github.com/max-models/byob/actions/workflows/static_analysis.yml)
[![Docs](https://github.com/max-models/byob/actions/workflows/docs.yml/badge.svg)](https://max-models.github.io/byob/)
[![codecov](https://codecov.io/gh/max-models/byob/branch/main/graph/badge.svg)](https://codecov.io/gh/max-models/byob)
[![PyPI](https://img.shields.io/pypi/v/byob.png)](https://pypi.org/project/byob/)
[![Python](https://img.shields.io/pypi/pyversions/byob.png)](https://pypi.org/project/byob/)

Build your own background: static plasma fields for particle tracking.
Give the density and temperatures at a few probe positions and the walls
(any [simgeometry](https://github.com/max-models/simgeometry) shape,
floating or biased); byob interpolates the probes over a grid, sets the
bulk potential from Boltzmann’s relation and adds the Debye sheath and
presheath at every wall from the local electron temperature.

Documentation: <https://max-models.github.io/byob/>

``` python
from simgeometry import Rectangle

from byob import Background, Probe, UniformGrid, Wall

grid = UniformGrid((0.0, 0.0), (0.05, 0.02), (51, 21))
probes = [
    Probe("density", (0.025, 0.01), 1e17),  # m^-3
    Probe("electron_temperature", (0.025, 0.01), 2.0),  # eV
    Probe("ion_temperature", (0.025, 0.01), 0.5),
]
background = Background.build(grid, probes, [Wall(Rectangle((-0.01, 0.01), (0.02, 0.04)))])

field = background.electric_field([[1e-5, 0.01]])  # at particle positions, V/m
maps = background.sample(resolution=200)  # xarray, for plasma-plots
```

## Install

With [uv](https://docs.astral.sh/uv/):

``` bash
make install    # uv sync --extra dev, plus the pre-commit hooks
uv run byob
```

Or create and activate a Python environment (3.10 or newer):

``` bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
```

Install the code and requirements with pip:

``` bash
pip install -e .
```

Run the code with:

``` bash
byob
```

The `test`, `docs` and `dev` extras install the test runner, the
documentation tooling and the linters:

``` bash
pip install -e ".[dev]"
```

## Development

Formatting and linting use [ruff](https://docs.astral.sh/ruff/), type
checking [pyright](https://microsoft.github.io/pyright/), run by
[pre-commit](https://pre-commit.com/) and in CI:

``` bash
make lint     # ruff check, ruff format --check, pyright
make test     # pytest with coverage
```

Commit messages follow [Conventional
Commits](https://www.conventionalcommits.org/); see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Build docs

The documentation in `docs/` is an Astro + Starlight site: hand-written
pages, the notebooks in `tutorials/` executed and published as pages,
complete Python scripts in `examples/` published with their plots, and
the API reference generated from the docstrings with
[starlight-pydocs](https://ewels.github.io/starlight-pydocs/). It needs
Node 22 or newer.

``` bash
make docs-install     # npm packages and the Python docs extra
make docs-notebooks   # execute tutorials/*.ipynb and convert them to pages
make docs-examples    # run examples/*.py and publish their plots
make docs-dev         # live preview at http://localhost:4321/byob/
make docs-build       # the static site in docs/dist
```

## Build the README

`README.md` is rendered from `README.qmd` with
[Quarto](https://quarto.org/):

``` bash
make readme
```

## Releases

Before merging a release to `main`, update the version in
`pyproject.toml`, `src/byob/__init__.py` and `CITATION.cff` (including
its release date), and add the release notes to `CHANGELOG.md`. The push
to `main` creates a GitHub release with a `vX.Y.Z` tag and publishes the
package to PyPI with trusted publishing (OIDC). The one-time PyPI and
GitHub configuration is described in the [publishing
tutorial](https://max-models.github.io/byob/development/publishing/).
