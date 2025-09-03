# PyGMT: Create Stunning Maps and Geospatial Visualizations in Python

**PyGMT is a Python interface for the Generic Mapping Tools (GMT), empowering you to generate publication-quality maps and figures with ease.** ([View the original repository](https://github.com/GenericMappingTools/pygmt))

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

*   **Pythonic Interface:** Access the power of GMT with a clean and intuitive Python API.
*   **Geospatial Data Processing:** Process and visualize geospatial and geophysical data.
*   **Publication-Quality Maps:** Create stunning figures and maps suitable for scientific publications.
*   **Integration with the Scientific Python Ecosystem:** Seamlessly integrates with NumPy, Pandas, xarray, and GeoPandas.
*   **Direct GMT C API Access:** Utilizes the GMT C API directly for efficient performance.
*   **Rich Display in Jupyter Notebooks:** Enhanced support for displaying maps and figures within Jupyter notebooks.

## Why Use PyGMT?

PyGMT simplifies the process of creating complex maps and visualizations, making it easier for scientists and researchers to communicate their findings effectively. Explore the power of PyGMT by trying it out online on [Binder](https://github.com/GenericMappingTools/try-gmt) or watch a quick introduction [here](https://youtu.be/4iPnITXrxVU).

## Getting Started

### Installation

Install PyGMT using [mamba](https://mamba.readthedocs.org/):

```bash
mamba install --channel conda-forge pygmt
```

Or with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html):

```bash
conda install --channel conda-forge pygmt
```

For more installation options, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Quick Example

Create a basic map in a Python interpreter or Jupyter notebook:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

Explore the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html) for more examples.

## Project Goals

*   Make GMT more accessible to new users.
*   Build a Pythonic API for GMT.
*   Interface with the GMT C API directly using ctypes (no system calls).
*   Support for rich display in the Jupyter notebook.
*   Integration with the [scientific Python ecosystem](https://scientific-python.org/): `numpy.ndarray` or `pandas.DataFrame` for data tables, `xarray.DataArray` for grids, and `geopandas.GeoDataFrame` for geographical data.

## Contact & Community

*   **Discussions:** Join the conversation on our [Discourse forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a) or open an issue on [GitHub](https://github.com/GenericMappingTools/pygmt/issues/new).

## Contributing

### Code of Conduct

This project adheres to the [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).

### Contributing Guidelines

Review our [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) to learn how you can contribute.

## Citing PyGMT

Please cite our work in your research:

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

For specific version citations, use the Zenodo page: <https://doi.org/10.5281/zenodo.3781524>.  It is also recommended to cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515).

## License

PyGMT is licensed under the [BSD 3-clause License](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

The development of PyGMT is supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

## Minimum Supported Versions

See [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) for detailed versioning information.