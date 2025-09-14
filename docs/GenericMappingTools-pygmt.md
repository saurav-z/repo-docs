# PyGMT: Create Stunning Maps and Figures with Python

**Unlock the power of the Generic Mapping Tools (GMT) directly from Python with PyGMT, a versatile library for geospatial data visualization and analysis.**  Check out the original repo: [https://github.com/GenericMappingTools/pygmt](https://github.com/GenericMappingTools/pygmt)

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

## Key Features of PyGMT:

*   **Pythonic Interface:** Easily access and utilize the powerful GMT command-line tools through a user-friendly Python API.
*   **Publication-Quality Maps:** Create stunning maps and figures for scientific publications and presentations.
*   **Geospatial Data Processing:** Process and visualize geospatial and geophysical data with ease.
*   **Jupyter Notebook Integration:** Seamlessly integrate with Jupyter notebooks for interactive data exploration and visualization.
*   **Scientific Python Ecosystem Compatibility:** Works with `numpy`, `pandas`, `xarray`, and `geopandas` for efficient data handling.
*   **Direct GMT Integration:**  Uses the GMT C API directly for optimized performance, without relying on system calls.
*   **Rich Gallery of Examples:** Explore a [comprehensive gallery](https://www.pygmt.org/latest/gallery) and [tutorials](https://www.pygmt.org/latest/tutorials) to get started quickly.

## Why Choose PyGMT?

PyGMT simplifies the process of generating complex maps and figures, enabling you to:

*   **Visualize complex geospatial data:** Transform your data into compelling visual narratives.
*   **Create publication-ready graphics:** Produce high-quality figures for scientific publications.
*   **Streamline your workflow:** Benefit from a Pythonic interface that simplifies GMT's functionality.
*   **Accelerate your research:**  Quickly explore and visualize your data with efficient processing and display.

Learn more by trying it online on [Binder](https://github.com/GenericMappingTools/try-gmt) or watching our [3 minute overview](https://youtu.be/4iPnITXrxVU)!

[![Quick Introduction to PyGMT YouTube Video](https://raw.githubusercontent.com/GenericMappingTools/pygmt/refs/heads/main/doc/_static/scipy2022-youtube-thumbnail.jpg)](https://www.youtube.com/watch?v=4iPnITXrxVU)

## Getting Started

### Installation

Install PyGMT using [mamba](https://mamba.readthedocs.io/):

```bash
mamba install --channel conda-forge pygmt
```

Alternatively, with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html):

```bash
conda install --channel conda-forge pygmt
```

For complete installation instructions, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Quick Example

Run the following code in a Python interpreter or [Jupyter notebook](https://docs.jupyter.org/en/latest/running.html) to get started:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

This will generate a global map.  Explore the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html) for more examples.

## Project Goals:

*   Make GMT more accessible to new users.
*   Build a Pythonic API for GMT.
*   Interface with the GMT C API directly using ctypes (no system calls).
*   Support for rich display in the Jupyter notebook.
*   Integration with the [scientific Python ecosystem](https://scientific-python.org/).

## Contact and Community

*   **GitHub:** [Open an issue](https://github.com/GenericMappingTools/pygmt/issues/new) or comment on existing ones.
*   **Discourse Forum:** Ask questions and leave comments on our [Discourse forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a).

## Contributing

We welcome your contributions!

*   **Code of Conduct:**  Please adhere to the [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).
*   **Contributing Guidelines:** See the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md).
*   **Imposter Syndrome Disclaimer:** We encourage everyone to contribute, regardless of experience.

## Citing PyGMT

To cite PyGMT in your research, use the following BibTeX:

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

Also, cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515).  See <https://www.generic-mapping-tools.org/cite> for additional citations.

## License

PyGMT is licensed under the **BSD 3-clause License**. See [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt) for details.

## Support

Development supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

## Minimum Supported Versions

See [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) for supported versions of GMT, Python and core package dependencies.