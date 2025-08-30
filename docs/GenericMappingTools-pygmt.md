# PyGMT: Create Stunning Maps and Geospatial Visualizations with Python

**PyGMT empowers you to generate publication-quality maps and figures with ease, directly from Python.** [Explore the PyGMT repository](https://github.com/GenericMappingTools/pygmt).

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

<!-- doc-index-start-after -->

## Key Features

*   **Pythonic Interface:** Seamlessly integrate with the power of GMT using a Python-friendly API.
*   **Geospatial Data Handling:** Process and visualize geospatial and geophysical data effortlessly.
*   **Publication-Quality Output:** Create stunning maps and figures suitable for publications and presentations.
*   **Integration with the Scientific Python Ecosystem:** Utilize `numpy`, `pandas`, `xarray`, and `geopandas` for data input and manipulation.
*   **Jupyter Notebook Support:** Enjoy rich display capabilities within Jupyter notebooks.

## Why PyGMT?

PyGMT provides a Pythonic interface to the powerful [Generic Mapping Tools (GMT)](https://github.com/GenericMappingTools/gmt), making complex geospatial data visualization accessible and efficient. Experience the power of PyGMT online with [try-gmt](https://github.com/GenericMappingTools/try-gmt)!

**Dive Deeper:**

*   [Tutorials](https://www.pygmt.org/latest/tutorials)
*   [Gallery](https://www.pygmt.org/latest/gallery)
*   [External PyGMT examples](https://www.pygmt.org/latest/external_resources.html)

[![Quick Introduction to PyGMT YouTube Video](https://raw.githubusercontent.com/GenericMappingTools/pygmt/refs/heads/main/doc/_static/scipy2022-youtube-thumbnail.jpg)](https://www.youtube.com/watch?v=4iPnITXrxVU)

## About

PyGMT is a Python library that provides a user-friendly interface for the [Generic Mapping Tools (GMT)](https://github.com/GenericMappingTools/gmt), a powerful command-line program widely used in Earth, Ocean, and Planetary sciences.

## Project Goals

*   Increase GMT's accessibility for new users.
*   Develop a Pythonic API for GMT.
*   Directly interface with the GMT C API via ctypes (no system calls).
*   Enhance rich display capabilities in Jupyter notebooks.
*   Integrate seamlessly with the scientific Python ecosystem.

## Quickstart

### Installation

Install PyGMT easily using [mamba](https://mamba.readthedocs.org/):

```bash
mamba install --channel conda-forge pygmt
```

Or, install with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html):

```bash
conda install --channel conda-forge pygmt
```

For comprehensive installation instructions, see the [full installation guide](https://www.pygmt.org/latest/install.html).

### Getting Started

Open a [Python interpreter](https://docs.python.org/3/tutorial/interpreter.html) or [Jupyter notebook](https://docs.jupyter.org/en/latest/running.html) and try this basic example:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

This will generate a global map.  Explore more examples in the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html).

## Contacting Us

*   Engage with the community on [GitHub](https://github.com/GenericMappingTools/pygmt) by [opening issues](https://github.com/GenericMappingTools/pygmt/issues/new) or commenting on existing ones.
*   Ask questions and share ideas on our [Discourse forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a).

## Contributing

### Code of Conduct

This project adheres to the [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).

### Contributing Guidelines

Review our [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) to learn how to contribute and provide valuable feedback.

### Imposter Syndrome Disclaimer

We welcome your contributions!  Remember, contributing goes beyond just code – it also involves documentation, testing, and providing feedback.

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

For specific versions, see the Zenodo page: <https://doi.org/10.5281/zenodo.3781524>. Also cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515). Additional citation information can be found at <https://www.generic-mapping-tools.org/cite>.

## License

PyGMT is licensed under the **BSD 3-clause License**. See [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

Development is supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

<!-- doc-index-end-before -->

## Minimum Supported Versions

PyGMT follows [SPEC 0](https://scientific-python.org/specs/spec-0000/) and has adopted extensions based on the project's requirements.  See [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) for details.