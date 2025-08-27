# PyGMT: Create Stunning Maps and Geospatial Visualizations with Python

**Unlock the power of the Generic Mapping Tools (GMT) directly within Python to create publication-quality maps and analyze geospatial data.** [Explore PyGMT on GitHub](https://github.com/GenericMappingTools/pygmt)

<!-- doc-index-start-after -->

## Key Features of PyGMT

*   **Pythonic Interface:** Seamlessly interact with GMT's powerful capabilities using an intuitive Python API.
*   **Geospatial Data Processing:** Process and visualize a wide range of geospatial data, including topography, bathymetry, and seismic data.
*   **Publication-Quality Maps:** Create stunning, customizable maps for scientific publications, presentations, and reports.
*   **Integration with Scientific Python Ecosystem:** Works smoothly with popular scientific Python libraries like NumPy, pandas, and xarray.
*   **Cross-Platform Compatibility:** Run PyGMT on Windows, macOS, and Linux.
*   **Direct GMT API Access:** Efficiently interfaces with the GMT C API for optimal performance.

## Why Use PyGMT?

PyGMT is the perfect solution for anyone who needs to create high-quality maps and visualizations from geospatial and geophysical data.  It provides an easy-to-learn Python interface to GMT. To truly understand how powerful PyGMT is, play with it online on [Binder](https://github.com/GenericMappingTools/try-gmt)! For a quicker introduction, check out our [3 minute overview](https://youtu.be/4iPnITXrxVU)!

Afterwards, feel free to look at our [Tutorials](https://www.pygmt.org/latest/tutorials), visit the [Gallery](https://www.pygmt.org/latest/gallery), and check out some [external PyGMT examples](https://www.pygmt.org/latest/external_resources.html)!

[![Quick Introduction to PyGMT YouTube Video](https://raw.githubusercontent.com/GenericMappingTools/pygmt/refs/heads/main/doc/_static/scipy2022-youtube-thumbnail.jpg)](https://www.youtube.com/watch?v=4iPnITXrxVU)

## Getting Started

### Installation

Install PyGMT easily using `mamba` or `conda`:

```bash
mamba install --channel conda-forge pygmt
```

```bash
conda install --channel conda-forge pygmt
```

For other ways to install `pygmt`, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Example

Get started quickly with this simple example:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

## Learn More

*   **Documentation:** [Development Version](https://www.pygmt.org/dev)
*   **Tutorials:**  Explore the [Tutorials](https://www.pygmt.org/latest/tutorials/index.html)
*   **Gallery:**  Browse the [Gallery](https://www.pygmt.org/latest/gallery/index.html)
*   **External Examples:** [External Resources](https://www.pygmt.org/latest/external_resources.html)
*   **Forum:**  [Discourse forum](https://forum.generic-mapping-tools.org)

## Contribute

We welcome contributions! Please review the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md).

### Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## Citing PyGMT

If you use PyGMT in your research, please cite it. See the [AUTHORS.md](https://github.com/GenericMappingTools/pygmt/blob/main/AUTHORS.md) file for a list of the people involved.

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

PyGMT is licensed under the [BSD 3-clause License](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt).

## Support

The development of PyGMT has been supported by NSF grants
[OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and
[EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

## Minimum Supported Versions

[Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) for the detailed policy and the minimum supported versions of GMT, Python and core package dependencies.

<!-- doc-index-end-before -->