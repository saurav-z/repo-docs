# PyGMT: Create Publication-Quality Maps with Python

**Visualize and analyze geospatial and geophysical data with ease using PyGMT, a powerful Python interface for the Generic Mapping Tools (GMT).** ([Back to GitHub Repo](https://github.com/GenericMappingTools/pygmt))

## Key Features of PyGMT

*   **Pythonic Interface:**  A user-friendly Python API that simplifies the use of GMT's powerful mapping capabilities.
*   **Geospatial Data Handling:**  Seamlessly integrates with the scientific Python ecosystem, supporting `numpy.ndarray`, `pandas.DataFrame`, `xarray.DataArray`, and `geopandas.GeoDataFrame`.
*   **Publication-Quality Maps:**  Create stunning, customizable maps and figures for your research.
*   **Direct GMT Integration:**  Leverages the GMT C API directly using `ctypes` for efficient performance (no system calls).
*   **Jupyter Notebook Support:**  Rich display capabilities within Jupyter notebooks for interactive exploration.
*   **Extensive Tutorials and Examples:**  Get started quickly with comprehensive documentation, tutorials, and a gallery of examples.

## Why Choose PyGMT?

PyGMT empowers scientists and researchers to generate publication-ready maps and figures for a wide range of applications.  Whether you're working in Earth science, oceanography, planetary science, or any field that uses geospatial data, PyGMT provides the tools you need.

*   **Ease of use:** Makes GMT more accessible to new users.
*   **Interoperability:** Integrates with the scientific Python ecosystem.
*   **Performance:**  Interfaces with the GMT C API for speed and efficiency.

## Getting Started

### Installation

Install PyGMT easily using [mamba](https://mamba.readthedocs.org/):

```bash
mamba install --channel conda-forge pygmt
```

Or using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html):

```bash
conda install --channel conda-forge pygmt
```

For detailed installation instructions, see the [full installation instructions](https://www.pygmt.org/latest/install.html).

### Quick Example

Create a global map in a few lines of code:

```python
import pygmt
fig = pygmt.Figure()
fig.coast(projection="N15c", region="g", frame=True, land="tan", water="lightblue")
fig.text(position="MC", text="PyGMT", font="80p,Helvetica-Bold,red@75")
fig.show()
```

## Resources

*   **Documentation:** [Development Version Documentation](https://www.pygmt.org/dev)
*   **Tutorials:** [PyGMT Tutorials](https://www.pygmt.org/latest/tutorials)
*   **Gallery:** [PyGMT Gallery](https://www.pygmt.org/latest/gallery)
*   **Quick Introduction:** [3 Minute Overview Video](https://youtu.be/4iPnITXrxVU)
*   **External Examples:** [External PyGMT Examples](https://www.pygmt.org/latest/external_resources.html)

## Contributing

We welcome contributions! Please review the [Contributing Guide](https://github.com/GenericMappingTools/pygmt/blob/main/CONTRIBUTING.md) and adhere to the [Contributor Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md).

## Contact and Support

*   **GitHub:** [GitHub Repository](https://github.com/GenericMappingTools/pygmt) - Open issues and contribute!
*   **Forum:** [Discourse Forum](https://forum.generic-mapping-tools.org/c/questions/pygmt-q-a) - Ask questions and discuss PyGMT.

## Citing PyGMT

If you use PyGMT in your research, please cite it. You can find the BibTeX citation and additional information on the [Zenodo page](https://doi.org/10.5281/zenodo.3781524). Also, cite the [GMT 6 paper](https://doi.org/10.1029/2019GC008515).

## License

PyGMT is released under the **BSD 3-clause License**.  See [LICENSE.txt](https://github.com/GenericMappingTools/pygmt/blob/main/LICENSE.txt) for details.

## Support

The development of PyGMT has been supported by NSF grants [OCE-1558403](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1558403) and [EAR-1948603](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1948602).

## Related Projects

*   [GMT.jl](https://github.com/GenericMappingTools/GMT.jl): A Julia wrapper for GMT.
*   [gmtmex](https://github.com/GenericMappingTools/gmtmex): A Matlab/Octave wrapper for GMT.

## Minimum Supported Versions

Please see [Minimum Supported Versions](https://www.pygmt.org/dev/minversions.html) for the detailed policy and the minimum supported versions of GMT, Python and core package dependencies.