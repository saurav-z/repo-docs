# PyGMT: Geospatial Data Visualization and Processing in Python

**Unlock the power of publication-quality maps and figures with PyGMT, a Python interface for the versatile Generic Mapping Tools (GMT).** Explore the original repository on [GitHub](https://github.com/GenericMappingTools/pygmt).

## Key Features

*   **Pythonic Interface:** Interact with GMT through an intuitive and easy-to-learn Python API.
*   **Geospatial Data Processing:** Process and visualize a wide range of geospatial and geophysical data.
*   **Publication-Quality Maps:** Create stunning maps and figures ready for scientific publications.
*   **Integration with the Scientific Python Ecosystem:** Seamlessly works with NumPy, Pandas, Xarray, and GeoPandas for data input.
*   **Direct C API Interface:** Utilizes a direct interface with the GMT C API using ctypes for performance.
*   **Jupyter Notebook Support:** Provides rich display capabilities within Jupyter notebooks.
*   **Versatile Data Handling:** Supports `numpy.ndarray`, `pandas.DataFrame`, `xarray.DataArray`, and `geopandas.GeoDataFrame`.

## About PyGMT

PyGMT is a powerful Python library built to empower scientists and researchers in creating high-quality maps and processing geospatial data. It offers a user-friendly Python interface for the widely-used Generic Mapping Tools (GMT), known for its command-line tools. PyGMT brings the power of GMT to the Python world, allowing you to visualize, analyze, and manipulate your data with ease.

## Project Goals

*   **Improved Accessibility:** Make GMT more accessible to both new and experienced users.
*   **Pythonic API:** Design a Pythonic API to ensure intuitive and efficient interactions.
*   **Direct C API Integration:** Interface directly with the GMT C API, enhancing performance through the use of ctypes.
*   **Enhanced Jupyter Support:** Offer rich display support within Jupyter notebooks.
*   **Ecosystem Integration:** Seamless integration with the scientific Python ecosystem, supporting data formats like `numpy.ndarray`, `pandas.DataFrame`, `xarray.DataArray`, and `geopandas.GeoDataFrame`.

## Quickstart

### Installation

Install PyGMT with [mamba](https://mamba.readthedocs.org/):

```bash
mamba install --channel conda-forge pygmt
```

Alternatively, use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html):

```bash
conda install --channel conda-forge pygmt
```

For more installation options, consult the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Getting Started

Get started by running a [Python interpreter](https://docs.python.org/3/tutorial/interpreter.html) or [Jupyter notebook](https://docs.jupyter.org/en/latest/running.html) and executing the following example:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

This will create a global map with colored land and water, as well as the text "PyGMT." For more detailed examples, check out the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html).

## Contact and Community

*   **GitHub:** Engage in discussions, report issues, and contribute through [GitHub](https://github.com/GenericMappingTools/pygmt).
*   **Discourse Forum:** Ask questions, share insights, and connect with the community on our [Discourse forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a).

## Contributing

Your contributions are welcome! Read the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) to learn how to contribute.

### Code of Conduct

This project adheres to the [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).

## Citing PyGMT

If you use PyGMT in your research, please cite the following BibTeX:

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

PyGMT is available under the [BSD 3-clause License](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

Development of PyGMT is supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

Explore other GMT wrappers:

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

## Minimum Supported Versions

PyGMT follows [SPEC 0](https://scientific-python.org/specs/spec-0000/) for minimum support and has extensions. For minimum supported versions, see [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html).