# PyGMT: Create Stunning Geospatial Maps and Figures with Python

**Unlock the power of the Generic Mapping Tools (GMT) directly from Python with PyGMT, enabling you to create publication-quality maps and figures with ease.** [Explore the PyGMT repository](https://github.com/GenericMappingTools/pygmt)

## Key Features of PyGMT

*   **Pythonic Interface:** Interact with GMT's powerful capabilities using a user-friendly Python API.
*   **Publication-Quality Maps:** Generate visually appealing maps and figures ready for scientific publications.
*   **Geospatial Data Processing:** Process and visualize a wide range of geospatial and geophysical data.
*   **Integration with Scientific Python Ecosystem:** Seamlessly works with NumPy, Pandas, xarray, and GeoPandas for data input and manipulation.
*   **Direct GMT API Access:** Leverages the GMT C API directly using ctypes for efficient performance.
*   **Jupyter Notebook Support:** Enjoy rich display and interactive capabilities within Jupyter notebooks.

## Why Choose PyGMT?

*   **Accessibility:** Make GMT more accessible to new users.
*   **Pythonic API:** A Pythonic API for GMT.
*   **Direct Interface:** Interface with the GMT C API directly using ctypes (no system calls).
*   **Jupyter Notebook:** Support for rich display in the Jupyter notebook.
*   **Ecosystem integration:** Integration with the [scientific Python ecosystem](https://scientific-python.org/).

## Installation

Install PyGMT using mamba or conda:

```bash
mamba install --channel conda-forge pygmt
# or
conda install --channel conda-forge pygmt
```

For detailed installation instructions, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

## Getting Started

Begin your PyGMT journey by running a simple example in a Python interpreter or Jupyter notebook:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

## Resources

*   **Documentation:** [Development version](https://www.pygmt.org/dev)
*   **Tutorials:** Explore our [tutorials](https://www.pygmt.org/latest/tutorials)
*   **Gallery:** See amazing examples in our [gallery](https://www.pygmt.org/latest/gallery)
*   **Quick Introduction:** [3-minute overview](https://youtu.be/4iPnITXrxVU)
*   **External examples:** [External PyGMT examples](https://www.pygmt.org/latest/external_resources.html)

## Contribute

We welcome contributions!  Please read our [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md).  Check out the [Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).

## Citing PyGMT

If you use PyGMT in your research, please cite it using the following BibTeX:

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
And it is strongly recommended to cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515).

## License

PyGMT is released under the **BSD 3-clause License**. See [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt) for details.

## Support

Development of PyGMT is supported by NSF grants
[OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and
[EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.