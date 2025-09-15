# PyGMT: Create Stunning Maps and Geospatial Visualizations with Python

**Unleash the power of the Generic Mapping Tools (GMT) directly within Python to craft publication-quality maps and analyze geospatial data.**  Find the original repo [here](https://github.com/GenericMappingTools/pygmt).

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

## Key Features of PyGMT

*   **Pythonic Interface:** Access GMT's powerful capabilities through an intuitive Python API.
*   **Publication-Quality Maps:** Create stunning, customizable maps and figures for scientific publications.
*   **Geospatial Data Processing:** Process and visualize geospatial and geophysical data with ease.
*   **Seamless Integration:** Works seamlessly with the scientific Python ecosystem (NumPy, Pandas, xarray, GeoPandas).
*   **Direct GMT C API Access:** Utilizes ctypes for direct interaction with the GMT C API, ensuring performance.
*   **Rich Display in Jupyter:** Supports rich display capabilities within Jupyter notebooks.
*   **Extensive Tutorials and Examples:** Benefit from comprehensive documentation, tutorials, and a gallery of examples to get started quickly.

## Why Choose PyGMT?

PyGMT empowers you to create high-quality maps and figures for your geospatial and geophysical research. It offers a user-friendly Python interface to the powerful GMT command-line tools, making complex data visualization tasks accessible.  Try it out online via [Binder](https://github.com/GenericMappingTools/try-gmt)!

## Getting Started

### Installation

Install PyGMT using `mamba` or `conda`:

```bash
mamba install --channel conda-forge pygmt
# OR
conda install --channel conda-forge pygmt
```

For detailed installation instructions, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Quickstart Example

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

This code generates a global map. Explore the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html) for more examples.

## Resources

*   [Documentation (development version)](https://www.pygmt.org/dev)
*   [Tutorials](https://www.pygmt.org/latest/tutorials)
*   [Gallery](https://www.pygmt.org/latest/gallery)
*   [External PyGMT examples](https://www.pygmt.org/latest/external_resources.html)
*   [3 minute overview](https://youtu.be/4iPnITXrxVU)

## Contact and Community

*   **GitHub:**  Engage in discussions, report issues, and contribute on [GitHub](https://github.com/GenericMappingTools/pygmt).
*   **Discourse Forum:** Ask questions and find support in the [Discourse forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a).

## Contributing

We welcome contributions! Please review the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) and adhere to our [Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).

## Citing PyGMT

If you use PyGMT in your research, please cite our work using the provided BibTeX information (available in the original README).

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

Also, please cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515). Find citation information for other modules at <https://www.generic-mapping-tools.org/cite>.

## License

PyGMT is licensed under the [BSD 3-clause License](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

The development of PyGMT is supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

<!-- doc-index-end-before -->

## Minimum Supported Versions

For details on minimum supported versions of GMT, Python, and dependencies, consult the [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) page.