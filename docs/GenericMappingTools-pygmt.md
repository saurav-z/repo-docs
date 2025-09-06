# PyGMT: Create Stunning Geospatial Visualizations with Python

**Transform your geospatial and geophysical data into publication-quality maps and figures with PyGMT, a powerful and user-friendly Python interface to the Generic Mapping Tools (GMT).** Access the original repo [here](https://github.com/GenericMappingTools/pygmt).

## Key Features

*   **Pythonic Interface:** A clean and intuitive Python API for interacting with GMT's extensive functionality.
*   **Publication-Quality Maps:** Generate visually appealing and scientifically accurate maps and figures.
*   **Geospatial Data Processing:** Process geospatial and geophysical data with ease.
*   **Integration with Scientific Python Ecosystem:** Seamlessly integrates with `numpy`, `pandas`, `xarray`, and `geopandas`.
*   **Direct GMT C API Access:** PyGMT utilizes the GMT C API directly, optimizing performance without relying on system calls.
*   **Jupyter Notebook Support:** Enjoy rich display capabilities within Jupyter notebooks for interactive exploration and visualization.

## Why Use PyGMT?

*   **Accessibility:** Easily create complex maps and figures without delving into command-line intricacies.
*   **Versatility:** Supports a wide range of mapping and data processing tasks.
*   **Reproducibility:** Script-based workflows ensure that your visualizations are easily reproducible.
*   **Community-Driven:** Benefit from an active and supportive community of developers and users.

## Getting Started

### Installation

Install PyGMT using `mamba` (recommended) or `conda`:

```bash
mamba install --channel conda-forge pygmt
```

Or using `conda`:

```bash
conda install --channel conda-forge pygmt
```

For other installation methods, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Basic Example

Get started with a simple example:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

This code produces a global map with land and water masses colored, and "PyGMT" overlaid. Explore the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html) for more examples.

## Resources

*   [Documentation](https://www.pygmt.org/dev)
*   [Tutorials](https://www.pygmt.org/latest/tutorials)
*   [Gallery](https://www.pygmt.org/latest/gallery)
*   [Contact](https://forum.generic-mapping-tools.org)
*   [TryOnline](https://github.com/GenericMappingTools/try-gmt)

## Contributing

We welcome contributions! Please review the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) and the [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).

## Citation

If you use PyGMT in your research, please cite it using the following BibTeX:

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

## License

PyGMT is released under the [BSD 3-clause License](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

Development of PyGMT is supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).