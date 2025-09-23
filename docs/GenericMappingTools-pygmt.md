# PyGMT: Geospatial Data Visualization and Analysis in Python

**Unlock the power of the Generic Mapping Tools (GMT) with PyGMT, a Pythonic interface for creating stunning maps and analyzing geospatial data.** Explore the original repository on [GitHub](https://github.com/GenericMappingTools/pygmt).

[![Latest version on PyPI](https://img.shields.io/pypi/v/pygmt)](https://pypi.org/project/pygmt)
[![Latest version on conda-forge](https://img.shields.io/conda/v/conda-forge/pygmt)](https://anaconda.org/conda-forge/pygmt)
[![GitHub license](https://img.shields.io/github/license/GenericMappingTools/pygmt)](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt)
[![Compatible Python versions](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FGenericMappingTools%2Fpygmt%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://www.pygmt.org/dev/minversions.html)
[![Digital Object Identifier for the Zenodo archive](https://zenodo.org/badge/DOI/10.5281/3781524.svg)](https://doi.org/10.5281/zenodo.3781524)
[![Discourse forum](https://img.shields.io/discourse/status?label=forum&server=https%3A%2F%2Fforum.generic-mapping-tools.org)](https://forum.generic-mapping-tools.org)
[![PyOpenSci](https://tinyurl.com/y22nb8up)](https://github.com/pyOpenSci/software-submission/issues/43)
[![Contributor Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg)](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md)
[![GitHub Actions Tests status](https://github.com/GenericMappingTools/pygmt/actions/workflows/ci_tests.yaml/badge.svg)](https://github.com/GenericMappingTools/pygmt/actions/workflows/ci_tests.yaml)
[![GitHub Actions GMT Dev Tests status](https://github.com/GenericMappingTools/pygmt/actions/workflows/ci_tests_dev.yaml/badge.svg)](https://github.com/GenericMappingTools/pygmt/actions/workflows/ci_tests_dev.yaml)
[![Test coverage status](https://codecov.io/gh/GenericMappingTools/pygmt/graph/badge.svg?token=78Fu4EWstx)](https://app.codecov.io/gh/GenericMappingTools/pygmt)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CodSpeed Performance Benchmarks](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/GenericMappingTools/pygmt)

## Key Features

*   **Pythonic Interface:**  Provides an intuitive Python interface to the powerful GMT command-line tools.
*   **Publication-Quality Maps:** Create stunning, customizable maps and figures for scientific publications.
*   **Geospatial Data Processing:** Process and visualize geospatial and geophysical data with ease.
*   **Integration with Python Ecosystem:** Seamlessly integrates with `numpy`, `pandas`, `xarray`, and `geopandas` for data handling.
*   **Interactive Visualization:** Supports rich display in Jupyter notebooks for immediate feedback and exploration.
*   **Direct GMT API Access:**  Interfaces directly with the GMT C API using `ctypes` for enhanced performance (without system calls).

## Why Use PyGMT?

PyGMT empowers you to transform complex geospatial data into compelling visualizations and insightful analyses, making it a vital tool for researchers, scientists, and anyone working with map data.  Get started quickly by exploring PyGMT with the [try-gmt](https://github.com/GenericMappingTools/try-gmt) Binder or a quick tour via the [3 minute overview](https://youtu.be/4iPnITXrxVU)!

## Getting Started

### Installation

Install PyGMT using `mamba` or `conda`:

```bash
mamba install --channel conda-forge pygmt
```

```bash
conda install --channel conda-forge pygmt
```

Or, see the [full installation instructions](https://www.pygmt.org/latest/install.html) for other installation options.

### Basic Example

Create a global map with land and water masses:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

## Resources

*   [Documentation](https://www.pygmt.org/dev)
*   [Tutorials](https://www.pygmt.org/latest/tutorials)
*   [Gallery](https://www.pygmt.org/latest/gallery)
*   [Contact](https://forum.generic-mapping-tools.org)
*   [External PyGMT examples](https://www.pygmt.org/latest/external_resources.html)

## Contributing

We welcome contributions!  Please review our [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) and [Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md) to get started.

## Citing PyGMT

Please cite PyGMT in your publications using the provided BibTeX entry found in the original README. It's also recommended to cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515)

## License

PyGMT is released under the BSD 3-clause License.  See [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt) for details.

## Support

Development of PyGMT is supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.