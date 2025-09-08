# PyGMT: Create Publication-Quality Maps and Figures with Python

**Transform your geospatial and geophysical data into stunning visuals with PyGMT, a powerful Python interface for the Generic Mapping Tools.**  [Explore the PyGMT Repository](https://github.com/GenericMappingTools/pygmt)

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

*   **Pythonic Interface:**  A user-friendly Python API for accessing the powerful capabilities of GMT.
*   **Publication-Quality Graphics:** Create stunning maps and figures for scientific publications and presentations.
*   **Geospatial Data Processing:**  Process and visualize geospatial and geophysical data with ease.
*   **Integration with Scientific Python Ecosystem:** Seamlessly integrates with `NumPy`, `Pandas`, `xarray`, and `GeoPandas` for efficient data handling.
*   **Direct GMT C API Access:**  Leverages the GMT C API directly for high performance (no system calls).
*   **Interactive Visualization:** Supports rich display of maps and figures within Jupyter notebooks.

## Why Choose PyGMT?

PyGMT is ideal for scientists, researchers, and anyone working with geospatial data who needs to create compelling visualizations.  It offers:

*   **Accessibility:** Easy to learn and use, making GMT accessible to both beginners and experienced users.
*   **Flexibility:**  A wide range of tools and options for customizing your maps and figures.
*   **Power:** The full power of GMT, a widely used and respected mapping and data processing tool.

## Getting Started

### Installation

Install PyGMT using `mamba` or `conda`:

```bash
mamba install --channel conda-forge pygmt
```

or

```bash
conda install --channel conda-forge pygmt
```

For detailed instructions, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Example

Get started quickly with a simple map:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

## Resources

*   [Documentation (development version)](https://www.pygmt.org/dev)
*   [Tutorials](https://www.pygmt.org/latest/tutorials)
*   [Gallery](https://www.pygmt.org/latest/gallery)
*   [External PyGMT examples](https://www.pygmt.org/latest/external_resources.html)
*   [Try PyGMT Online](https://github.com/GenericMappingTools/try-gmt)

## Contact and Community

*   [Discourse Forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a)
*   [GitHub](https://github.com/GenericMappingTools/pygmt)

## Contributing

We welcome contributions!  Please review the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) and the [Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).

## Citing PyGMT

Please cite our work in your research using the following BibTeX:

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

Also, cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515) and any specific module citations. See <https://www.generic-mapping-tools.org/cite> for details.

## License

PyGMT is licensed under the **BSD 3-clause License**. A copy of the license is available in [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

Development is supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

## Minimum Supported Versions

See [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) for the detailed policy.