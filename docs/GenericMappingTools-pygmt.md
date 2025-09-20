# PyGMT: Effortlessly Create Stunning Maps and Geospatial Visualizations with Python

**PyGMT empowers you to generate publication-quality maps and figures from geospatial and geophysical data using Python, built on the powerful Generic Mapping Tools (GMT).**  [Explore the PyGMT Repository](https://github.com/GenericMappingTools/pygmt)

## Key Features

*   **Pythonic Interface:** A user-friendly Python API simplifies interaction with GMT's powerful capabilities.
*   **Publication-Quality Output:** Generate professional-grade maps and figures for scientific publications, presentations, and reports.
*   **Geospatial Data Processing:** Easily process and visualize a wide range of geospatial and geophysical data.
*   **Integration with Scientific Python Ecosystem:** Seamlessly works with `NumPy`, `pandas`, `xarray`, and `GeoPandas` for efficient data handling.
*   **Interactive Visualization:** Supports rich display in Jupyter notebooks for immediate feedback and exploration.
*   **Command-line Tool Access:** Offers the full capabilities of GMT's command-line tools for advanced customization.

## About PyGMT

PyGMT is a Python library providing an intuitive interface to the Generic Mapping Tools (GMT). Designed for scientists and researchers, it allows you to create compelling visualizations for diverse fields like Earth Sciences, Oceanography, and Planetary Science. Whether you are a beginner or an experienced user, PyGMT offers a flexible and powerful way to analyze geospatial data and create high-quality maps.  Dive deeper into how PyGMT works with the [TryOnline](https://github.com/GenericMappingTools/try-gmt) and the [3 minute overview](https://youtu.be/4iPnITXrxVU).

## Getting Started

### Installation

Install PyGMT using [mamba](https://mamba.readthedocs.org/):

```bash
mamba install --channel conda-forge pygmt
```

or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html):

```bash
conda install --channel conda-forge pygmt
```

For detailed installation instructions, visit the [PyGMT installation guide](https://www.pygmt.org/latest/install.html).

### Simple Example

Get started quickly with this basic example:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

Explore further examples and tutorials in the [Gallery](https://www.pygmt.org/latest/gallery) and [Tutorials](https://www.pygmt.org/latest/tutorials).

## Resources

*   **Documentation:** [PyGMT Documentation (development version)](https://www.pygmt.org/dev)
*   **Forum:** [GMT Forum](https://forum.generic-mapping-tools.org)
*   **Examples:** [External PyGMT Examples](https://www.pygmt.org/latest/external_resources.html)
*   **Quick Introduction to PyGMT YouTube Video:** [![Quick Introduction to PyGMT YouTube Video](https://raw.githubusercontent.com/GenericMappingTools/pygmt/refs/heads/main/doc/_static/scipy2022-youtube-thumbnail.jpg)](https://www.youtube.com/watch?v=4iPnITXrxVU)

## Contributing & Community

*   **Join the Community:** Engage with the PyGMT community on [GitHub](https://github.com/GenericMappingTools/pygmt) and the [Discourse forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a).
*   **Contribute:** Learn how to contribute to the project through the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md).
*   **Code of Conduct:**  This project is governed by a [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).

## Citing PyGMT

If you use PyGMT in your research, please cite the project using the following BibTeX entry, which includes a DOI:

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

Also, remember to cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515).

## License

PyGMT is distributed under the [BSD 3-clause License](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support & Related Projects

*   **Support:** The development of PyGMT is supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).
*   **Related Projects:**
    *   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
    *   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.