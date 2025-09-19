# PyGMT: Create Stunning Maps and Geospatial Visualizations in Python

**Unlock the power of the Generic Mapping Tools (GMT) with PyGMT, a user-friendly Python interface for creating publication-quality maps and analyzing geospatial data.** ([View on GitHub](https://github.com/GenericMappingTools/pygmt))

## Key Features:

*   **Pythonic Interface:** Easy-to-learn and use Python API for GMT functionality.
*   **Publication-Quality Maps:** Generate professional-grade maps and figures.
*   **Geospatial Data Processing:** Process and visualize geospatial and geophysical data.
*   **Integration with Scientific Python Ecosystem:** Seamlessly works with NumPy, Pandas, xarray, and GeoPandas.
*   **Direct GMT API Access:** Utilizes the GMT C API directly for performance.
*   **Jupyter Notebook Support:** Rich display and interactive capabilities within Jupyter notebooks.
*   **Extensive Documentation & Examples:** Comprehensive documentation, tutorials, and a gallery to get you started.
*   **Community-Driven:** Active community with a forum and open-source contributions.

## About PyGMT

PyGMT is a Python library designed for geospatial data processing, analysis, and visualization. It provides a user-friendly Pythonic interface to the powerful [Generic Mapping Tools (GMT)](https://github.com/GenericMappingTools/gmt), a command-line program widely used across Earth, Ocean, and Planetary sciences, and beyond. PyGMT empowers you to create publication-ready maps and figures with ease. Explore the [3-minute overview](https://youtu.be/4iPnITXrxVU) to understand how easy it is to create powerful maps with PyGMT.

## Why Use PyGMT?

*   **Simplified GMT Access:** PyGMT makes the complexities of GMT accessible through a Pythonic interface.
*   **Rapid Prototyping:** Quickly create and iterate on your maps and visualizations.
*   **Reproducible Research:** Benefit from the power of code to ensure your results are reproducible.
*   **Publication-Ready Output:** Generate high-quality maps and figures for your publications.
*   **Integration with Existing Workflows:** Seamlessly incorporate PyGMT into your Python-based scientific workflows.

## Getting Started

### Installation

Install PyGMT using [mamba](https://mamba.readthedocs.org/):

```bash
mamba install --channel conda-forge pygmt
```

Or, using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html):

```bash
conda install --channel conda-forge pygmt
```

Refer to the [full installation instructions](https://www.pygmt.org/latest/install.html) for other installation options.

### Example

Create a beautiful map using PyGMT:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

Explore the [Gallery](https://www.pygmt.org/latest/gallery/index.html) and [Tutorials](https://www.pygmt.org/latest/tutorials/index.html) for more examples.

## Resources

*   **Documentation:** [Development Version](https://www.pygmt.org/dev)
*   **Tutorials:** [Tutorials](https://www.pygmt.org/latest/tutorials)
*   **Gallery:** [Gallery](https://www.pygmt.org/latest/gallery)
*   **External Examples:** [External PyGMT Examples](https://www.pygmt.org/latest/external_resources.html)
*   **YouTube Intro:** [Quick Introduction to PyGMT](https://youtu.be/4iPnITXrxVU)

## Community & Support

*   **Forum:** [Discourse forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a)
*   **GitHub:** [Open an Issue](https://github.com/GenericMappingTools/pygmt/issues/new)

## Contributing

We welcome contributions! Read our [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) to get involved.

*   **Code of Conduct:**  Please adhere to our [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).

## Citing PyGMT

Please cite PyGMT in your publications using the following BibTeX:

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

Also, please cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515) if you are using PyGMT.

## License

PyGMT is licensed under the **BSD 3-clause License**. See [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt) for details.

## Support

The development of PyGMT has been supported by NSF grants
[OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and
[EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).