# PyGMT: Pythonic Mapping and Geospatial Data Visualization

**Create stunning, publication-quality maps and analyze geospatial data effortlessly with PyGMT, a Python interface for the powerful Generic Mapping Tools.**  [Learn more at the PyGMT GitHub repository.](https://github.com/GenericMappingTools/pygmt)

## Key Features of PyGMT

*   **Pythonic Interface:** A user-friendly Python API that simplifies the use of the Generic Mapping Tools (GMT).
*   **Publication-Quality Mapping:** Generate professional-grade maps and figures for scientific publications.
*   **Geospatial Data Processing:** Process and analyze geospatial and geophysical data.
*   **Integration with the Scientific Python Ecosystem:** Seamlessly works with NumPy, Pandas, xarray, and GeoPandas for data input and manipulation.
*   **Jupyter Notebook Compatibility:** Rich display and interactive plotting support within Jupyter notebooks.
*   **Direct GMT C API Access:** Interfaces with the GMT C API directly using ctypes for optimal performance.

## Why Choose PyGMT?

*   **Accessibility:** Makes GMT more accessible to new users by providing a Pythonic API.
*   **Versatility:** Suitable for a wide range of applications, including Earth science, oceanography, planetary science, and more.
*   **Community Driven:** Benefit from an active and supportive community of scientists and developers.

## Installation

Get started quickly with `mamba` or `conda`:

```bash
mamba install --channel conda-forge pygmt
```

or

```bash
conda install --channel conda-forge pygmt
```

For other installation methods, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

## Getting Started

Here's a quick example to get you started:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

Explore the [Gallery](https://www.pygmt.org/latest/gallery) and [Tutorials](https://www.pygmt.org/latest/tutorials) for more in-depth examples and tutorials.

## Resources

*   **Documentation:** [Development Version Documentation](https://www.pygmt.org/dev)
*   **Contact:** [Forum](https://forum.generic-mapping-tools.org)
*   **Try Online:** [Try PyGMT Online](https://github.com/GenericMappingTools/try-gmt)

## Support and Community

*   **GitHub:** [Open an issue](https://github.com/GenericMappingTools/pygmt/issues/new) or comment on existing ones.
*   **Discourse Forum:** [Ask questions and share feedback](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a).

## Contributing

We welcome contributions! Please review the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) for details.

## Citing PyGMT

If you use PyGMT in your research, please cite the project using the following BibTeX:

```bibtex
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

PyGMT is licensed under the [BSD 3-clause License](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

---
<!-- Badges and Additional Information (Keep as is) -->
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