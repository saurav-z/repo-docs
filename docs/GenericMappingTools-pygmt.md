# PyGMT: Create Stunning Maps and Geospatial Visualizations with Python

**PyGMT empowers scientists and researchers to visualize geospatial and geophysical data and create publication-quality maps directly from Python.**  [Learn more at the original repository](https://github.com/GenericMappingTools/pygmt).

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

*   **Pythonic Interface:**  Offers an intuitive Python API for the powerful Generic Mapping Tools (GMT).
*   **Publication-Quality Maps:** Create visually appealing maps and figures for scientific publications.
*   **Geospatial Data Processing:** Process and visualize geospatial and geophysical data with ease.
*   **Integration with the Scientific Python Ecosystem:** Seamlessly works with `numpy`, `pandas`, `xarray`, and `geopandas`.
*   **Direct GMT API Access:** Interfaces directly with the GMT C API using `ctypes` for optimal performance.
*   **Jupyter Notebook Support:** Rich display features within Jupyter notebooks.

## Getting Started

### Installation

Install PyGMT easily with:

```bash
mamba install --channel conda-forge pygmt
```

or using conda:

```bash
conda install --channel conda-forge pygmt
```

For other installation options, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Quick Example

Create a basic map:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

Explore more examples in the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html).

## Why PyGMT?

PyGMT simplifies the use of GMT, making it accessible for new users and providing a modern Pythonic interface.  Visualize geospatial data with ease, create publication-quality maps, and integrate with the scientific Python ecosystem.
Watch the 3 minute overview:  [3 minute overview](https://youtu.be/4iPnITXrxVU)!

## Project Goals

*   Make GMT more accessible to new users.
*   Build a Pythonic API for GMT.
*   Interface with the GMT C API directly using ctypes (no system calls).
*   Support for rich display in the Jupyter notebook.
*   Integration with the [scientific Python ecosystem](https://scientific-python.org/): `numpy.ndarray` or `pandas.DataFrame` for data tables, `xarray.DataArray` for grids, and `geopandas.GeoDataFrame` for geographical data.

## Contact and Community

*   **GitHub:** [Open an issue](https://github.com/GenericMappingTools/pygmt/issues/new) or comment on existing issues.
*   **Discourse Forum:** [Questions and Discussion](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a)

## Contributing

We welcome contributions! Please review our:

*   [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md)
*   [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md)

## Citing PyGMT

If you use PyGMT in your research, please cite:

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

Also, cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515).

## License

PyGMT is available under the **BSD 3-clause License**.  See [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

Development supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

## Minimum Supported Versions

See the [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) page for the minimum supported versions of GMT, Python, and core dependencies.