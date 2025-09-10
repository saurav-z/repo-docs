# PyGMT: Pythonic Geospatial Mapping and Data Visualization

**Transform your geospatial and geophysical data into stunning, publication-quality maps with PyGMT, the Python interface for the powerful Generic Mapping Tools.**  [Visit the PyGMT GitHub Repository](https://github.com/GenericMappingTools/pygmt)

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

*   **Pythonic Interface:**  A user-friendly Python API for interacting with the powerful GMT command-line tools.
*   **Publication-Quality Maps:**  Create stunning and customizable maps and figures for scientific publications and presentations.
*   **Geospatial Data Processing:** Comprehensive tools for handling and visualizing geospatial and geophysical data.
*   **Integration with Scientific Python Ecosystem:** Seamlessly integrates with `NumPy`, `pandas`, `xarray`, and `geopandas` for data input and output.
*   **Jupyter Notebook Compatibility:** Rich display support for interactive plotting within Jupyter notebooks.
*   **Cross-Platform Compatibility**: Runs on all major operating systems (Linux, macOS, Windows).

## Why PyGMT?

PyGMT simplifies the process of creating complex maps and data visualizations, making it easier for scientists and researchers to communicate their findings effectively.  Explore its capabilities with the [online Binder](https://github.com/GenericMappingTools/try-gmt) or dive in with the [3-minute overview video](https://youtu.be/4iPnITXrxVU)!  Then explore the [tutorials](https://www.pygmt.org/latest/tutorials), [gallery](https://www.pygmt.org/latest/gallery), and [examples](https://www.pygmt.org/latest/external_resources.html).

[![Quick Introduction to PyGMT YouTube Video](https://raw.githubusercontent.com/GenericMappingTools/pygmt/refs/heads/main/doc/_static/scipy2022-youtube-thumbnail.jpg)](https://www.youtube.com/watch?v=4iPnITXrxVU)

## About

PyGMT is a Python library providing a convenient interface to the [Generic Mapping Tools (GMT)](https://github.com/GenericMappingTools/gmt), a powerful command-line suite widely used in Earth, Ocean, and Planetary sciences. This library makes it easier to process geospatial and geophysical data and create high-quality figures.

## Project Goals

*   Improve GMT accessibility for new users.
*   Develop a Pythonic API for GMT.
*   Interface with the GMT C API directly using `ctypes` (no system calls).
*   Support rich display in Jupyter notebooks.
*   Integrate with the scientific Python ecosystem.

## Quickstart

### Installation

Install PyGMT using `mamba` or `conda`:

```bash
mamba install --channel conda-forge pygmt
```

or

```bash
conda install --channel conda-forge pygmt
```

For other installation methods, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Getting Started

Open a Python interpreter or Jupyter notebook and run the following example:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

This will produce a global map.  Explore the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html) for more examples.

## Contact and Community

*   Discuss and contribute [on GitHub](https://github.com/GenericMappingTools/pygmt), or [open an issue](https://github.com/GenericMappingTools/pygmt/issues/new).
*   Ask questions on our [Discourse forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a).

## Contributing

We welcome your contributions!  Please review our [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md).

### Code of Conduct

This project adheres to a [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).

### Imposter Syndrome Disclaimer

We encourage all potential contributors to participate, regardless of experience level.

## Citing PyGMT

If you use PyGMT in your research, please cite the software using the following BibTeX entry and also the GMT 6 paper:

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

Get the most up-to-date citation information from the [Zenodo page](https://doi.org/10.5281/zenodo.3781524).  Also cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515).

## License

PyGMT is released under the [BSD 3-clause License](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

Development of PyGMT has been supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

<!-- doc-index-end-before -->

## Minimum Supported Versions

Refer to [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) for details.