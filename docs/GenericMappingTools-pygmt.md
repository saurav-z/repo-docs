# PyGMT: Create Stunning Maps and Geospatial Visualizations in Python

**Transform your geospatial data into publication-quality maps and figures with PyGMT, a powerful Python interface for the Generic Mapping Tools (GMT).** Explore the original repository [here](https://github.com/GenericMappingTools/pygmt).

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

*   **Pythonic Interface:**  A user-friendly Pythonic API for the powerful GMT command-line tools.
*   **Publication-Quality Graphics:** Create professional-grade maps and figures for scientific publications and presentations.
*   **Geospatial Data Processing:** Process and visualize geospatial and geophysical data with ease.
*   **Integration with Scientific Python Ecosystem:** Seamlessly integrates with `numpy`, `pandas`, `xarray`, and `geopandas` for data handling.
*   **Interactive Visualization:** Rich display support in Jupyter notebooks.
*   **Direct C API Access:**  Interfaces with the GMT C API directly using `ctypes`, avoiding system calls for improved performance.
*   **Extensive Functionality:** Access a vast range of GMT modules for mapping, data processing, and analysis.

## What is PyGMT?

PyGMT is a Python library that provides a Pythonic interface for the [Generic Mapping Tools (GMT)](https://github.com/GenericMappingTools/gmt). GMT is a widely used command-line program for creating high-quality maps and figures, particularly within the Earth, Ocean, and Planetary sciences. PyGMT simplifies the use of GMT by providing a user-friendly Python API, making it easier for both new and experienced users to create stunning visualizations.

## Benefits of Using PyGMT

*   **Accessibility:** Makes GMT more accessible to new users, especially those familiar with Python.
*   **Productivity:**  Simplifies the process of creating complex maps and figures.
*   **Flexibility:**  Offers a wide range of options for customizing maps and visualizations.
*   **Reproducibility:** Ensures reproducibility of your results through code.

## Getting Started

### Installation

Install PyGMT using `mamba` (recommended) or `conda`:

```bash
mamba install --channel conda-forge pygmt
# or
conda install --channel conda-forge pygmt
```

For alternative installation methods, consult the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Quick Example

Get started quickly by running this example in a Python interpreter or Jupyter notebook:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

For more detailed examples and tutorials, explore the [Gallery](https://www.pygmt.org/latest/gallery) and [Tutorials](https://www.pygmt.org/latest/tutorials).

## Contact & Support

*   **GitHub:** Engage in discussions and report issues on [GitHub](https://github.com/GenericMappingTools/pygmt).
*   **Discourse Forum:** Ask questions and interact with the community on the [Discourse forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a).

## Contributing

We welcome contributions!  See our [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) for details.

## Citing PyGMT

If you use PyGMT in your research, please cite the project using the following BibTeX entry:

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

Remember to also cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515). For further information, see  <https://www.generic-mapping-tools.org/cite>.

## License

PyGMT is distributed under the **BSD 3-clause License**.  See [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt) for details.

## Support

Development is supported by NSF grants:  [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

Explore other official GMT wrappers:

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

## Minimum Supported Versions

PyGMT follows [SPEC 0](https://scientific-python.org/specs/spec-0000/) and related extensions. See [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) for detailed version information.