# PyGMT: Create Stunning Maps & Geospatial Visualizations with Python

**Unlock the power of the Generic Mapping Tools (GMT) within Python and effortlessly create publication-quality maps and figures for your geospatial and geophysical data.**  [Explore the PyGMT Repository](https://github.com/GenericMappingTools/pygmt)

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

*   **Pythonic Interface:**  Provides an intuitive and user-friendly Python API for GMT, making it easier to learn and use.
*   **Publication-Quality Maps:**  Create professional-grade maps and figures for scientific publications and presentations.
*   **Geospatial Data Processing:**  Process and visualize geospatial and geophysical data, including seismic, bathymetry, and more.
*   **Integration with the Scientific Python Ecosystem:**  Seamlessly integrates with popular Python libraries like NumPy, Pandas, and xarray for data manipulation.
*   **Interactive Visualization:**  Supports rich display within Jupyter notebooks for interactive exploration and analysis.
*   **Direct GMT C API Access:** Utilizes ctypes to directly interface with the GMT C API, optimizing performance without system calls.

## Why Use PyGMT?

PyGMT empowers you to harness the power of GMT's extensive functionality within the Python environment. It simplifies complex mapping tasks, allows you to create high-quality visualizations, and facilitates geospatial data analysis, making it ideal for researchers, scientists, and anyone working with geographic data. Get started today!  Explore the [3 minute overview](https://youtu.be/4iPnITXrxVU)!

## Getting Started

### Installation

Easily install PyGMT using [mamba](https://mamba.readthedocs.org/):

```bash
mamba install --channel conda-forge pygmt
```

Or with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html):

```bash
conda install --channel conda-forge pygmt
```

For detailed installation instructions, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Example

Create a global map in seconds:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

For more examples, explore the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html).

## Resources

*   **Documentation:** [PyGMT Documentation](https://www.pygmt.org/dev)
*   **Tutorials:** [PyGMT Tutorials](https://www.pygmt.org/latest/tutorials)
*   **Gallery:** [PyGMT Gallery](https://www.pygmt.org/latest/gallery)
*   **External Resources:** [External PyGMT Examples](https://www.pygmt.org/latest/external_resources.html)

## Contact and Community

*   **Discussions:** [GitHub Issues](https://github.com/GenericMappingTools/pygmt/issues/new)
*   **Forum:** [Discourse Forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a)

## Contributing

We welcome contributions! Please review our:

*   **Code of Conduct:** [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md)
*   **Contribution Guidelines:** [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md)

## Citing PyGMT

If you use PyGMT in your research, please cite us:

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

Also cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515). More citation information can be found at <https://www.generic-mapping-tools.org/cite>.

## License

PyGMT is licensed under the **BSD 3-clause License**.  See [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

Development is supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

## Minimum Supported Versions

See [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) for details.