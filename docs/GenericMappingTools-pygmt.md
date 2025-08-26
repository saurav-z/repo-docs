# PyGMT: Create Stunning Maps and Geospatial Visualizations in Python

**Transform your geospatial data into publication-quality maps with the power of Python and the Generic Mapping Tools (GMT)!** [Visit the PyGMT GitHub Repository](https://github.com/GenericMappingTools/pygmt)

PyGMT is a Python interface for the powerful and versatile Generic Mapping Tools (GMT), providing a user-friendly way to process geospatial and geophysical data and create stunning visualizations. It's ideal for scientists, researchers, and anyone needing to visualize geographic information effectively.

## Key Features:

*   **Pythonic Interface:**  Enjoy a Python-native API for GMT, making it easier to learn and use.
*   **Publication-Quality Maps:** Generate high-resolution, publication-ready maps and figures.
*   **Geospatial Data Processing:**  Process and visualize a wide array of geospatial data, including gridded datasets, vector data, and more.
*   **Integration with Scientific Python Ecosystem:** Seamlessly integrates with NumPy, Pandas, xarray, and GeoPandas for efficient data handling.
*   **Rich Display in Jupyter Notebook:**  Visualize your maps directly within Jupyter notebooks for interactive exploration and analysis.
*   **Cross-Platform Compatibility:**  Works on various operating systems, including macOS, Linux, and Windows.
*   **Active Community & Support:** Benefit from a supportive community and extensive documentation.
*   **Open Source:** PyGMT is free and open-source, licensed under the BSD 3-clause License.

## Getting Started

### Installation

Install PyGMT using [mamba](https://mamba.readthedocs.org/):

```bash
mamba install --channel conda-forge pygmt
```

or with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html):

```bash
conda install --channel conda-forge pygmt
```

For detailed installation instructions, see the [PyGMT installation guide](https://www.pygmt.org/latest/install.html).

### Example Usage

Quickly create a global map:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

## Resources

*   [Documentation (Development Version)](https://www.pygmt.org/dev)
*   [Tutorials](https://www.pygmt.org/latest/tutorials)
*   [Gallery](https://www.pygmt.org/latest/gallery)
*   [Quick Introduction Video](https://youtu.be/4iPnITXrxVU)
*   [External Examples](https://www.pygmt.org/latest/external_resources.html)

## Contributing

We welcome contributions!  Please review our [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) for details on how to get involved.  We also have a [Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md) in place.

## Support & Community

*   [GitHub Issues](https://github.com/GenericMappingTools/pygmt/issues/new):  Report issues, ask questions, and contribute.
*   [Discourse Forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a):  Get help and discuss PyGMT with the community.

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
Remember to also cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515).