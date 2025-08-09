# VisiData: Explore and Arrange Tabular Data in Your Terminal

**Tired of struggling with spreadsheets and command-line tools?** VisiData provides a powerful and intuitive terminal interface for exploring, analyzing, and manipulating tabular data directly from your terminal. 

[Visit the original repository on GitHub](https://github.com/saulpw/visidata)

## Key Features of VisiData

*   **Versatile Data Format Support:**
    *   Supports a wide range of formats, including TSV, CSV, SQLite, JSON, XLSX (Excel), HDF5, and many more ([see all formats](https://visidata.org/formats)).
*   **Intuitive Interface:** Navigate and manipulate your data using familiar keyboard shortcuts.
*   **Powerful Commands:**  Utilize hundreds of commands for filtering, sorting, joining, and transforming your data.
*   **Cross-Platform Compatibility:** Works seamlessly on Linux, macOS, and Windows (with WSL).
*   **Flexible Installation:** Install easily via pip or from the development branch.

## Getting Started

### System Requirements

*   Linux, OS/X, or Windows (with WSL)
*   Python 3.8+
*   Additional Python modules may be needed for specific formats and sources.

### Installation

Install the latest stable release:

```bash
pip3 install visidata
```

Or, to install the development version:

```bash
pip3 install git+https://github.com/saulpw/visidata.git@develop
```

For comprehensive installation instructions, including platform-specific details and package manager options, see the official documentation: [visidata.org/install](https://visidata.org/install)

### Usage

To start exploring your data:

```bash
vd <input_file>
```

You can also pipe data into VisiData:

```bash
<command> | vd
```

Quit at any time with `Ctrl+Q`.

## Documentation and Resources

*   **Comprehensive Documentation:** [VisiData Documentation](https://visidata.org/docs)
*   **API Reference:** [Plugin Author's Guide and API Reference](https://visidata.org/docs/api)
*   **Quick Reference:** Access a command and options list within VisiData with `Ctrl+H`.
*   **Tutorial:** [Intro to VisiData Tutorial](https://jsvine.github.io/intro-to-visidata/) by [Jeremy Singer-Vine](https://www.jsvine.com/)

## Support and Community

*   **GitHub Issues:**  Report bugs, ask questions, and suggest features: [Create an issue](https://github.com/saulpw/visidata/issues)
*   **Discord:** Join the VisiData community chat: [visidata.org/chat](https://visidata.org/chat)

## License

VisiData is released under the GPLv3 license.

## Credits

VisiData is developed by Saul Pwanson.  Documentation and packaging are maintained by Anja Kefala.  Thanks to numerous contributors and users for their support.