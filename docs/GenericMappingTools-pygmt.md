# PyGMT: Create Stunning Maps and Geospatial Visualizations with Python

**PyGMT empowers you to generate publication-quality maps and figures for Earth, Ocean, and Planetary sciences using a Pythonic interface.** Learn more and contribute at the [original PyGMT repository](https://github.com/GenericMappingTools/pygmt).

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

*   **Pythonic Interface:**  Enjoy a user-friendly Python API for the powerful [Generic Mapping Tools (GMT)](https://github.com/GenericMappingTools/gmt).
*   **Geospatial Data Processing:**  Process and visualize geospatial and geophysical data with ease.
*   **Publication-Quality Graphics:** Create stunning maps and figures for scientific publications, presentations, and reports.
*   **Direct GMT API Access:**  Interact with the GMT C API directly via `ctypes` for efficient performance.
*   **Jupyter Notebook Integration:**  Seamlessly display your maps and figures within Jupyter notebooks.
*   **Integration with the Scientific Python Ecosystem:**  Works seamlessly with `NumPy`, `Pandas`, `xarray`, and `GeoPandas` for data handling.

## Getting Started

### Installation

Install PyGMT using [mamba](https://mamba.readthedocs.org/):

```bash
mamba install --channel conda-forge pygmt
```

Or using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html):

```bash
conda install --channel conda-forge pygmt
```

For other installation methods, refer to the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Example

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

This code will generate a global map. Explore more examples in the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html).

## Why Use PyGMT?

PyGMT simplifies the process of creating complex maps and visualizations. Check out this [3-minute video overview](https://youtu.be/4iPnITXrxVU)!

## Project Goals

*   Make GMT more accessible to new users.
*   Build a Pythonic API for GMT.
*   Interface with the GMT C API directly using ctypes (no system calls).
*   Support for rich display in the Jupyter notebook.
*   Integration with the [scientific Python ecosystem](https://scientific-python.org/):
  `numpy.ndarray` or `pandas.DataFrame` for data tables, `xarray.DataArray` for grids,
  and `geopandas.GeoDataFrame` for geographical data.

## Contact and Community

*   Discuss on [GitHub](https://github.com/GenericMappingTools/pygmt) - open an [issue](https://github.com/GenericMappingTools/pygmt/issues/new).
*   Ask questions on the [Discourse forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a).

## Contributing

*   Review the [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).
*   Read the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) for contribution guidelines.

### Imposter Syndrome Disclaimer

We encourage and value all contributions. If you want to contribute, please see the disclaimer in the original README.

## Citing PyGMT

Cite our work in your research using the following BibTeX:

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

Also, cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515). More info at <https://www.generic-mapping-tools.org/cite>.

## License

PyGMT is licensed under the [BSD 3-clause License](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

Development supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

## Minimum Supported Versions

See [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) for details on supported versions of GMT, Python, and dependencies.