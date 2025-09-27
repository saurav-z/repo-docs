# PyGMT: Create stunning maps and figures with a Pythonic interface to the Generic Mapping Tools

**PyGMT provides a user-friendly Python interface for the powerful Generic Mapping Tools (GMT), empowering you to create publication-quality maps and figures with ease.** [Visit the PyGMT Repository](https://github.com/GenericMappingTools/pygmt).

## Key Features

*   **Pythonic Interface:** Access the full power of GMT through an intuitive and Python-friendly API.
*   **Geospatial Data Processing:** Process and visualize geospatial and geophysical data.
*   **Publication-Quality Output:** Generate high-resolution maps and figures suitable for scientific publications.
*   **Integration with the Scientific Python Ecosystem:** Works seamlessly with `NumPy`, `Pandas`, `xarray`, and `GeoPandas` for data input and manipulation.
*   **Interactive Visualization:** Create rich, interactive visualizations in Jupyter notebooks.
*   **Direct GMT C API Access:** Efficiently interacts with the GMT C API using `ctypes`, avoiding system calls.
*   **Rich Documentation and Examples:** Access comprehensive documentation, tutorials, and a gallery of examples to get you started quickly.

## What is PyGMT?

PyGMT is a Python library that serves as a bridge between the versatile command-line GMT and the Python environment. It enables users to harness the full potential of GMT for geospatial data processing and visualization while leveraging the convenience and power of Python.

## Project Goals

*   Make GMT more accessible to both new and experienced users.
*   Offer a Pythonic API for easy interaction with GMT.
*   Use the GMT C API directly using ctypes for maximum efficiency.
*   Provide strong integration with the scientific Python ecosystem.

## Getting Started

### Installation

Install PyGMT using `mamba` or `conda`:

```bash
mamba install --channel conda-forge pygmt
# or
conda install --channel conda-forge pygmt
```

For detailed installation instructions, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Basic Example

Create a global map with a simple example:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

Explore the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html) for more in-depth examples.

## Support and Community

*   **Discussions:** [GitHub Issues](https://github.com/GenericMappingTools/pygmt/issues/new) for bug reports and feature requests.
*   **Forum:** [Discourse forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a) for general questions.

## Contribute

Join the PyGMT community and contribute to the project!  See the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) for more information.

## Citing PyGMT

Please cite PyGMT in your research using the following BibTeX:

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

It is also recommended to cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515).

## License

PyGMT is released under the **BSD 3-clause License**.  A copy of the license is available in [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).