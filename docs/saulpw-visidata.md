# VisiData: Explore and Transform Tabular Data in Your Terminal

**Effortlessly analyze and manipulate your data with VisiData, a powerful terminal-based interface that brings spreadsheet-like functionality to your command line.**  Learn more about this amazing tool on the [original repository](https://github.com/saulpw/visidata).

[![Tests](https://github.com/saulpw/visidata/workflows/visidata-ci-build/badge.svg)](https://github.com/saulpw/visidata/actions/workflows/main.yml)
[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/saulpw/visidata)
[![discord](https://img.shields.io/discord/880915750007750737?label=discord)](https://visidata.org/chat)
[![mastodon @visidata@fosstodon.org][2.1]][2]
[![twitter @VisiData][1.1]][1]

![Frequency table](http://visidata.org/freq-move-row.gif)

## Key Features

*   **Versatile Data Format Support:**  Open and work with a wide array of formats, including TSV, CSV, SQLite, JSON, XLSX (Excel), HDF5, and many more.  See a complete list at [visidata.org/formats](https://visidata.org/formats).
*   **Terminal-Based Interface:**  Explore and manipulate data directly in your terminal for efficient workflows.
*   **Intuitive Navigation & Commands:**  Utilize familiar spreadsheet-like commands and keyboard shortcuts.
*   **Interactive Data Exploration:**  Quickly filter, sort, and analyze your data within the terminal environment.
*   **Extensible with Plugins:**  Extend VisiData's capabilities with custom plugins.
*   **Cross-Platform Compatibility:** Works on Linux, macOS, and Windows (with WSL).

## Installation

To get started, install VisiData using `pip3`:

```bash
pip3 install visidata
```

For the cutting-edge `develop` branch:

```bash
pip3 install git+https://github.com/saulpw/visidata.git@develop
```

Detailed installation instructions are available at [visidata.org/install](https://visidata.org/install).

## Usage

Run VisiData with a file or pipe data into it:

```bash
vd <input_file>
<command> | vd
```

Press `Ctrl+Q` at any time to quit.  Discover hundreds of commands and options within the documentation.

## Documentation and Resources

*   [VisiData Documentation](https://visidata.org/docs)
*   [Plugin Author's Guide and API Reference](https://visidata.org/docs/api)
*   [Quick Reference](https://visidata.org/man) (accessible within `vd` with `Ctrl+H`)
*   [Intro to VisiData Tutorial](https://jsvine.github.io/intro-to-visidata/) by [Jeremy Singer-Vine](https://www.jsvine.com/)

## Support and Community

*   For questions, issues, or suggestions, please [create an issue on Github](https://github.com/saulpw/visidata/issues).
*   Join the VisiData community on [irc.libera.chat](https://libera.chat/) at #visidata.
*   Consider supporting the project on [Patreon](https://www.patreon.com/saulpw).

## License

VisiData is available under the GPLv3 license.

## Credits

VisiData is developed by Saul Pwanson `<vd@saul.pw>`. Documentation and packaging are maintained by Anja Kefala `<anja.kefala@gmail.com>`.  Many thanks to all contributors and users.  See the full list at [visidata.org/credits](https://visidata.org/credits/).

[1.1]: http://i.imgur.com/tXSoThF.png
[1]: http://www.twitter.com/VisiData
[2.1]: https://raw.githubusercontent.com/mastodon/mastodon/main/app/javascript/images/logo.svg
[2]: https://fosstodon.org/@visidata