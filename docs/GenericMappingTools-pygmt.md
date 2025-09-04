# PyGMT: Geospatial Data Processing and Mapping in Python

**Create stunning, publication-quality maps and figures with ease using PyGMT, the Python interface to the powerful Generic Mapping Tools.** Explore the original repository on [GitHub](https://github.com/GenericMappingTools/pygmt).

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

## Key Features of PyGMT

*   **Pythonic Interface:** Easy-to-use Python API for the powerful GMT command-line tools.
*   **Geospatial Data Processing:** Process and visualize geospatial and geophysical data.
*   **Publication-Quality Maps:** Create stunning maps and figures for scientific publications.
*   **Integration with Scientific Python Ecosystem:** Seamlessly integrates with NumPy, Pandas, and xarray for data handling.
*   **Jupyter Notebook Support:** Rich display and interactive plotting within Jupyter notebooks.
*   **Extensive Gallery & Tutorials:** Explore a wealth of examples and resources to get you started.

## Why PyGMT?

PyGMT empowers you to transform complex geospatial data into informative and visually appealing maps. Experience the power of PyGMT with a hands-on try on [Binder](https://github.com/GenericMappingTools/try-gmt) or watch a [3 minute overview](https://youtu.be/4iPnITXrxVU)!

## About

PyGMT provides a user-friendly Python interface to the [Generic Mapping Tools (GMT)](https://github.com/GenericMappingTools/gmt), a widely-used suite of command-line programs for geospatial data processing and visualization. It is used across the Earth, Ocean, and Planetary sciences, as well as other fields.

## Project Goals

*   Enhance accessibility of GMT for new users.
*   Develop a Pythonic API for GMT functionality.
*   Directly interface with the GMT C API via ctypes for efficient operation.
*   Offer rich display capabilities within Jupyter notebooks.
*   Integrate with the [scientific Python ecosystem](https://scientific-python.org/).

## Quickstart

### Installation

Install PyGMT using [mamba](https://mamba.readthedocs.org/):

```bash
mamba install --channel conda-forge pygmt
```

Alternatively, using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html):

```bash
conda install --channel conda-forge pygmt
```

For further installation options, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Getting Started

Here's a basic example to get you started:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

More examples and tutorials are available in the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html).

## Contact and Community

*   [GitHub](https://github.com/GenericMappingTools/pygmt) for discussion and issue tracking.
*   [Discourse forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a) for questions and community support.

## Contributing

*   [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md)
*   [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md)

## Citing PyGMT

Please cite PyGMT in your research using the following BibTeX:

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

Also, cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515). Find additional citation information at <https://www.generic-mapping-tools.org/cite>.

## License

PyGMT is licensed under the **BSD 3-clause License**. See [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt) for details.

## Support

Development supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

## Minimum Supported Versions

See [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) for details on supported versions of GMT, Python, and dependencies.