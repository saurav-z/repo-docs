# PyGMT: Pythonic Mapping and Data Visualization with the Generic Mapping Tools

**Create publication-quality maps and analyze geospatial data with ease using PyGMT, a powerful Python interface for the renowned Generic Mapping Tools (GMT).**  [Visit the original repository](https://github.com/GenericMappingTools/pygmt) for more information.

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

*   **Pythonic Interface:**  Access the powerful GMT functionalities through an intuitive Python API.
*   **Publication-Quality Maps:** Generate stunning, customizable maps and figures for scientific publications.
*   **Geospatial Data Processing:**  Process and analyze geospatial and geophysical data with a wide range of tools.
*   **Data Compatibility:** Seamlessly integrates with popular scientific Python libraries like NumPy, Pandas, and xarray.
*   **Rich Display:** Supports rich display within Jupyter notebooks.
*   **Easy Installation:** Install using conda or pip.

## Why Use PyGMT?

PyGMT simplifies the process of creating complex maps and visualizations. Whether you're a seasoned geoscientist or a beginner, PyGMT empowers you to:

*   Quickly create publication-ready figures.
*   Explore and analyze geospatial data efficiently.
*   Leverage the robust capabilities of GMT within a Pythonic environment.

To truly understand how powerful PyGMT is, play with it online on [Binder](https://github.com/GenericMappingTools/try-gmt)! For a
quicker introduction, check out our [3 minute overview](https://youtu.be/4iPnITXrxVU)!

Afterwards, feel free to look at our [Tutorials](https://www.pygmt.org/latest/tutorials),
visit the [Gallery](https://www.pygmt.org/latest/gallery), and check out some
[external PyGMT examples](https://www.pygmt.org/latest/external_resources.html)!

[![Quick Introduction to PyGMT YouTube Video](https://raw.githubusercontent.com/GenericMappingTools/pygmt/refs/heads/main/doc/_static/scipy2022-youtube-thumbnail.jpg)](https://www.youtube.com/watch?v=4iPnITXrxVU)

## Getting Started

### Installation

Install using mamba or conda:

```bash
mamba install --channel conda-forge pygmt
```

or

```bash
conda install --channel conda-forge pygmt
```

See the [full installation instructions](https://www.pygmt.org/latest/install.html) for other options.

### Example

Get started by running a Python interpreter or Jupyter notebook:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

For more examples, see the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and
[Tutorials](https://www.pygmt.org/latest/tutorials/index.html).

## Contact and Community

*   **GitHub:** [Open an issue](https://github.com/GenericMappingTools/pygmt/issues/new) or comment on any open issue.
*   **Discourse Forum:**  [Ask questions and leave comments](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a).

## Contributing

We welcome contributions!  Please review the following resources:

*   **Code of Conduct:**  Adhering to the [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md) is required.
*   **Contributing Guidelines:**  Learn how to contribute in the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md).

## Citing PyGMT

If you use PyGMT in your research, please cite us using the following BibTeX:

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

## License

PyGMT is licensed under the [BSD 3-clause License](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

Development of PyGMT is supported by NSF grants
[OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and
[EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

## Minimum Supported Versions

See [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) for details on supported versions of GMT, Python, and dependencies.