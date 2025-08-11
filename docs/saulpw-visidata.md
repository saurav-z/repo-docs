# VisiData: Explore and Manipulate Tabular Data in Your Terminal

**Effortlessly explore, analyze, and transform your data directly within your terminal with VisiData, a powerful and versatile data exploration tool.** 

[View the VisiData Repository on GitHub](https://github.com/saulpw/visidata)

## Key Features of VisiData

*   **Versatile Data Format Support:** Open and work with a wide variety of formats including CSV, TSV, JSON, XLSX (Excel), SQLite, HDF5, and many more ([see supported formats](https://visidata.org/formats)).
*   **Interactive Data Exploration:** Navigate, filter, sort, and group your data with an intuitive terminal interface.
*   **Powerful Command System:** Utilize hundreds of commands and options for in-depth data manipulation and analysis.
*   **Cross-Platform Compatibility:** Works seamlessly on Linux, macOS, and Windows (with WSL).
*   **Extensive Documentation:** Access comprehensive documentation, tutorials, and a quick reference guide (accessible within `vd` via `Ctrl+H`).
*   **Extensible with Plugins:** Extend functionality with custom plugins.

## Installation

### Using pip

To install the latest release from PyPi:

```bash
pip3 install visidata
```

To install the cutting-edge `develop` branch (use with caution):

```bash
pip3 install git+https://github.com/saulpw/visidata.git@develop
```

**For detailed installation instructions for all platforms and package managers, please visit [visidata.org/install](https://visidata.org/install).**

## Usage

Simply launch VisiData with a data source or pipe data to it:

```bash
vd <input_file.csv>
<command> | vd
```

Quit at any time with `Ctrl+Q`.

## Documentation and Resources

*   [VisiData Documentation](https://visidata.org/docs)
*   [Plugin Author's Guide and API Reference](https://visidata.org/docs/api)
*   [Quick Reference](https://visidata.org/man) (within `vd` with `Ctrl+H`)
*   [Intro to VisiData Tutorial](https://jsvine.github.io/intro-to-visidata/) by [Jeremy Singer-Vine](https://www.jsvine.com/)

## Get Help and Support

*   **Report Issues or Suggest Features:** [Create an issue on Github](https://github.com/saulpw/visidata/issues)
*   **Chat with the Community:** Join us on #visidata on [irc.libera.chat](https://libera.chat/).

## Support the Project

If you find VisiData valuable, consider supporting the project on [Patreon](https://www.patreon.com/saulpw)!

## License

VisiData is available under the GPLv3 license.

## Credits

*   **Lead Developer:** Saul Pwanson `<vd@saul.pw>`
*   **Documentation & Packaging:** Anja Kefala `<anja.kefala@gmail.com>`

A big thank you to all [contributors](https://visidata.org/credits/) and users!