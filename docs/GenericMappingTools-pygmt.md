# PyGMT: Create Stunning Geospatial Maps with Python

**PyGMT is a powerful Python library that allows you to create publication-quality maps and figures from geospatial and geophysical data, empowering you to visualize and analyze complex data with ease.**  [Explore the PyGMT Repository](https://github.com/GenericMappingTools/pygmt)

## Key Features:

*   **Pythonic Interface:**  Provides an intuitive and Python-friendly way to interact with the powerful Generic Mapping Tools (GMT) command-line programs.
*   **Publication-Quality Maps:** Generate stunning maps and figures with highly customizable options for visualization.
*   **Geospatial Data Processing:** Easily process and visualize geospatial and geophysical datasets.
*   **Integration with Scientific Python Ecosystem:** Seamlessly integrates with NumPy, Pandas, xarray, and GeoPandas for data handling.
*   **Direct GMT API Access:** Uses the GMT C API directly via ctypes, eliminating system calls for improved performance.
*   **Interactive Visualization:** Supports rich display and interaction within Jupyter notebooks.
*   **Extensive Documentation & Examples:**  Access comprehensive documentation, tutorials, and a gallery of examples to get started quickly.

## What is PyGMT?

PyGMT is a Python library that serves as an interface for the Generic Mapping Tools (GMT), offering a streamlined approach to geospatial and geophysical data processing and visualization. It is an essential tool for scientists and researchers in Earth sciences, oceanography, and related fields.

## Project Goals:

*   Enhance GMT accessibility for new users.
*   Develop a Pythonic API for GMT.
*   Directly interface with the GMT C API, ensuring efficient data processing.
*   Provide rich display capabilities within the Jupyter notebook environment.
*   Integrate smoothly with the core scientific Python ecosystem (NumPy, Pandas, xarray, GeoPandas).

## Getting Started

### Installation

Install PyGMT easily using `mamba` or `conda`:

```bash
mamba install --channel conda-forge pygmt
```

Or with `conda`:

```bash
conda install --channel conda-forge pygmt
```

For detailed installation instructions, please refer to the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Example

Start by importing the `pygmt` library and create your first map:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

Explore the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html) for more examples.

## Resources

*   **Documentation:** [Development Version Documentation](https://www.pygmt.org/dev)
*   **Forum:** [PyGMT Discourse Forum](https://forum.generic-mapping-tools.org)
*   **Try PyGMT Online:** [Try PyGMT Online on Binder](https://github.com/GenericMappingTools/try-gmt)

## Contribution & Community

*   **Contribute:**  Read the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md)
*   **Code of Conduct:**  Adheres to the [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).
*   **Discussion:** [GitHub Issues](https://github.com/GenericMappingTools/pygmt/issues/new)

## Citing PyGMT

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

For specific versions and GMT citations, consult the Zenodo page and the [GMT 6 paper](https://doi.org/10.1029/2019GC008515).

## License

PyGMT is distributed under the **BSD 3-clause License**. See [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

The development of PyGMT has been supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.