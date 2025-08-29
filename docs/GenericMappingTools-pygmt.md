# PyGMT: Create Stunning Maps & Geospatial Visualizations with Python

**Unlock the power of the Generic Mapping Tools (GMT) directly within Python and create publication-quality maps and figures with ease using PyGMT.**  [Explore the PyGMT repository](https://github.com/GenericMappingTools/pygmt).

---

## Key Features of PyGMT:

*   **Pythonic Interface:** A user-friendly Python API for GMT, making it easier for new users to get started.
*   **Publication-Quality Maps:** Generate professional-grade maps and figures for scientific publications and presentations.
*   **Geospatial Data Processing:** Process and visualize geospatial and geophysical data with powerful GMT tools.
*   **Direct C API Integration:**  Interfaces directly with the GMT C API for optimized performance (no system calls).
*   **Interactive Visualization:** Supports rich display in Jupyter notebooks for easy data exploration and map creation.
*   **Ecosystem Compatibility:** Seamlessly integrates with the scientific Python ecosystem, including NumPy, Pandas, Xarray, and GeoPandas.
*   **Broad Functionality:** Access a vast array of GMT modules for mapping, data processing, and analysis.

---

## Why Use PyGMT?

PyGMT is your gateway to creating compelling maps and visualizations. Whether you're an Earth scientist, oceanographer, or planetary scientist, PyGMT empowers you to:

*   **Visualize complex geospatial data effectively.**
*   **Create high-quality figures for publications.**
*   **Automate mapping workflows.**
*   **Explore and analyze data interactively.**

See it in action on [TryOnline](https://github.com/GenericMappingTools/try-gmt)!

[![Quick Introduction to PyGMT YouTube Video](https://raw.githubusercontent.com/GenericMappingTools/pygmt/refs/heads/main/doc/_static/scipy2022-youtube-thumbnail.jpg)](https://www.youtube.com/watch?v=4iPnITXrxVU)

## Getting Started

### Installation

Install PyGMT easily using [Mamba](https://mamba.readthedocs.io/):

```bash
mamba install --channel conda-forge pygmt
```

Or, with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html):

```bash
conda install --channel conda-forge pygmt
```

For other installation options, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Example

Get started quickly in a Python interpreter or Jupyter notebook:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

---

## Resources

*   [Documentation (development version)](https://www.pygmt.org/dev)
*   [Tutorials](https://www.pygmt.org/latest/tutorials)
*   [Gallery](https://www.pygmt.org/latest/gallery)
*   [External PyGMT Examples](https://www.pygmt.org/latest/external_resources.html)
*   [Contact](https://forum.generic-mapping-tools.org)

---

## Contributing

*   [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md)
*   [Contributing Guidelines](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md)

## Citing PyGMT

Please cite PyGMT and GMT in your research.  Detailed citation information can be found in the original README (link at the top).

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

PyGMT is released under the [BSD 3-clause License](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

Development is supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.