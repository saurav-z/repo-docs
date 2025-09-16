# PyGMT: Create Stunning Maps and Geospatial Visualizations in Python

**Visualize your geospatial data with ease using PyGMT, a powerful Python interface for the renowned Generic Mapping Tools (GMT).**  [Explore the original PyGMT repository](https://github.com/GenericMappingTools/pygmt).

## Key Features

*   **Pythonic Interface:** Seamlessly interact with GMT using a Python-friendly API.
*   **Publication-Quality Maps:** Generate professional-grade maps and figures for scientific publications.
*   **Geospatial Data Processing:** Process and visualize geospatial and geophysical data with ease.
*   **Integration with Scientific Python:** Works smoothly with `numpy`, `pandas`, `xarray`, and `geopandas`.
*   **Interactive Visualization:** Supports rich display within Jupyter notebooks.
*   **Direct C API Access:** Uses the GMT C API directly via `ctypes` for optimal performance.
*   **Wide Range of Applications:** Ideal for Earth sciences, oceanography, planetary sciences, and beyond.

## Why Choose PyGMT?

PyGMT empowers you to create compelling visualizations, making complex data accessible and impactful. Get started today by playing with it online on [Binder](https://github.com/GenericMappingTools/try-gmt) or view this [3 minute overview](https://youtu.be/4iPnITXrxVU).

## Core Functionality

*   [Tutorials](https://www.pygmt.org/latest/tutorials)
*   [Gallery](https://www.pygmt.org/latest/gallery)
*   [External PyGMT examples](https://www.pygmt.org/latest/external_resources.html)

[![Quick Introduction to PyGMT YouTube Video](https://raw.githubusercontent.com/GenericMappingTools/pygmt/refs/heads/main/doc/_static/scipy2022-youtube-thumbnail.jpg)](https://www.youtube.com/watch?v=4iPnITXrxVU)

## Installation

### Using mamba:

```bash
mamba install --channel conda-forge pygmt
```

### Using conda:

```bash
conda install --channel conda-forge pygmt
```

For comprehensive installation options, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

## Quickstart

Get started by creating a map in your [Python interpreter](https://docs.python.org/3/tutorial/interpreter.html) or a [Jupyter notebook](https://docs.jupyter.org/en/latest/running.html)

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

## Resources & Support

*   **Documentation:** [Development version](https://www.pygmt.org/dev)
*   **Forum:** [Discourse forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a)
*   **GitHub:** [Issues](https://github.com/GenericMappingTools/pygmt/issues/new)

## Contributing

We welcome contributions!  Please review the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) and [Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).

## Citing PyGMT

Please cite PyGMT in your research using the provided BibTeX.  Refer to the [AUTHORS.md](https://github.com/GenericMappingTools/pygmt/blob/main/AUTHORS.md) file for contributors.

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

Also, cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515).
## License

PyGMT is released under the [BSD 3-clause License](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

Development of PyGMT is supported by NSF grants
[OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and
[EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

## Minimum Supported Versions
See [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html)