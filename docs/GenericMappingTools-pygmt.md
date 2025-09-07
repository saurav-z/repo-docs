# PyGMT: Create Stunning Geospatial Visualizations with Python

**Transform your geospatial data into publication-quality maps and figures with PyGMT, a powerful Python interface for the Generic Mapping Tools (GMT).**  [See the original repo](https://github.com/GenericMappingTools/pygmt).

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

*   **Pythonic Interface:** Access the power of GMT with an intuitive Python API.
*   **Publication-Quality Graphics:** Generate professional maps and figures for scientific publications.
*   **Geospatial Data Processing:** Process and visualize geospatial and geophysical data.
*   **Integration with the Scientific Python Ecosystem:**  Seamlessly works with `numpy`, `pandas`, `xarray`, and `geopandas`.
*   **Interactive Visualization:** Create rich displays in Jupyter notebooks.
*   **Direct C API Interaction:**  Leverages the GMT C API directly using ctypes for efficiency.

## Why Choose PyGMT?

PyGMT empowers Earth scientists, oceanographers, and anyone working with geospatial data to create stunning visualizations. For a quick introduction, watch our [3-minute overview](https://youtu.be/4iPnITXrxVU) or explore the interactive [Binder](https://github.com/GenericMappingTools/try-gmt). Learn more with our [tutorials](https://www.pygmt.org/latest/tutorials) and explore the [gallery](https://www.pygmt.org/latest/gallery).

## Getting Started

### Installation

Install PyGMT using [mamba](https://mamba.readthedocs.org/):

```bash
mamba install --channel conda-forge pygmt
```

Or, using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html):

```bash
conda install --channel conda-forge pygmt
```

For other installation methods, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Example

Start by opening a [Python interpreter](https://docs.python.org/3/tutorial/interpreter.html) or a [Jupyter notebook](https://docs.jupyter.org/en/latest/running.html), and try this example:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

This will generate a global map with land, water and the text "PyGMT". See the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html) for more examples.

## Project Goals

*   Make GMT more accessible to new users.
*   Build a Pythonic API for GMT.
*   Interface with the GMT C API directly using ctypes (no system calls).
*   Support for rich display in the Jupyter notebook.
*   Integration with the [scientific Python ecosystem](https://scientific-python.org/): `numpy.ndarray` or `pandas.DataFrame` for data tables, `xarray.DataArray` for grids, and `geopandas.GeoDataFrame` for geographical data.

## Contact & Community

*   **GitHub:** [Open an issue](https://github.com/GenericMappingTools/pygmt/issues/new) or comment on existing ones.
*   **Discourse Forum:** [Ask questions and discuss](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a).

## Contributing

We welcome your contributions!  Please review the [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md) and our [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md).

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

Also, remember to cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515).

## License

PyGMT is licensed under the **BSD 3-clause License**.  See [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt) for details.

## Support

Development is supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl):  Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex):  Matlab/Octave wrapper for GMT.

## Minimum Supported Versions

See the [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) for details on GMT, Python, and package dependencies.