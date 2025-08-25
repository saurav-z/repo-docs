# PyGMT: Create Stunning Geospatial Maps and Figures with Python

**Transform your geospatial and geophysical data into publication-quality visualizations with PyGMT, a powerful Python interface for the renowned Generic Mapping Tools (GMT).** Explore the original repository at [https://github.com/GenericMappingTools/pygmt](https://github.com/GenericMappingTools/pygmt).

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

*   **Pythonic Interface:** Easily access the power of GMT through a user-friendly Python API.
*   **Geospatial Data Processing:** Process and visualize a wide range of geospatial and geophysical data.
*   **Publication-Quality Maps:** Create stunning maps and figures for scientific publications, presentations, and reports.
*   **Integration with the Scientific Python Ecosystem:** Seamlessly works with NumPy arrays, Pandas DataFrames, xarray Grids, and GeoPandas GeoDataFrames.
*   **Jupyter Notebook Support:** Rich display support for interactive exploration and visualization within Jupyter notebooks.
*   **Direct GMT C API access:** Fast and performant, utilizing the GMT C API directly without system calls.

## Why Use PyGMT?

PyGMT empowers researchers, scientists, and data analysts to create compelling visualizations for understanding and communicating complex geospatial data. Get started today by playing with it online on [Binder](https://github.com/GenericMappingTools/try-gmt) or watch the [3 minute overview](https://youtu.be/4iPnITXrxVU).

Explore our [Tutorials](https://www.pygmt.org/latest/tutorials), browse the [Gallery](https://www.pygmt.org/latest/gallery), and check out some [external PyGMT examples](https://www.pygmt.org/latest/external_resources.html)!

[![Quick Introduction to PyGMT YouTube Video](https://raw.githubusercontent.com/GenericMappingTools/pygmt/refs/heads/main/doc/_static/scipy2022-youtube-thumbnail.jpg)](https://www.youtube.com/watch?v=4iPnITXrxVU)

## About PyGMT

PyGMT is a Python library built on top of the Generic Mapping Tools (GMT), a powerful command-line program. It provides a Pythonic interface to GMT's functionalities, allowing you to process geospatial and geophysical data, and generate publication-quality maps and figures.

## Project Goals

*   Increase GMT accessibility for new users.
*   Develop a Pythonic API for GMT.
*   Interface directly with the GMT C API using ctypes.
*   Support rich display within Jupyter notebooks.
*   Integrate with the scientific Python ecosystem (NumPy, Pandas, xarray, GeoPandas).

## Quickstart

### Installation

Install PyGMT using [mamba](https://mamba.readthedocs.org/):

```bash
mamba install --channel conda-forge pygmt
```

Or, using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html):

```bash
conda install --channel conda-forge pygmt
```

See the [full installation instructions](https://www.pygmt.org/latest/install.html) for other installation methods.

### Getting Started

Open a [Python interpreter](https://docs.python.org/3/tutorial/interpreter.html) or a [Jupyter notebook](https://docs.jupyter.org/en/latest/running.html), and run this example:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

This will generate a global map with land and water masses, and the "PyGMT" text. Explore the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html) for more examples.

## Contact and Support

*   [GitHub](https://github.com/GenericMappingTools/pygmt): Open an issue or comment on existing issues.
*   [Discourse forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a): Ask questions and leave comments.

## Contributing

*   **Code of Conduct:** [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).
*   **Contributing Guidelines:** Read the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) for guidance.
*   **Imposter Syndrome Disclaimer:** We welcome your contributions!

## Citing PyGMT

Cite PyGMT in your research:

```
@software{
  pygmt_2025_15628725,
  author       = {Tian, Dongdong and
                  Uieda, Leonardo and
                  Leong, Wei Ji and
                  Fröhlich, Yvonne and
                  Grund, Michael and
                  Schlitzer, William and
                  Jones, Max and
                  Toney, Liam and
                  Yao, Jiayuan and
                  Tong, Jing-Hui and
                  Magen, Yohai and
                  Materna, Kathryn and
                  Belem, Andre and
                  Newton, Tyler and
                  Anant, Abhishek and
                  Ziebarth, Malte and
                  Quinn, Jamie and
                  Wessel, Paul},
  title        = {{PyGMT: A Python interface for the Generic Mapping Tools}},
  month        = jun,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.16.0},
  doi          = {10.5281/zenodo.15628725},
  url          = {https://doi.org/10.5281/zenodo.15628725}
}
```

Also, cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515). Find more info at [https://www.generic-mapping-tools.org/cite](https://www.generic-mapping-tools.org/cite).

## License

PyGMT is licensed under the **BSD 3-clause License**. See [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

Development supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

## Minimum Supported Versions

See [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) for details.