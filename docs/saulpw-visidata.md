# VisiData: Explore and Arrange Tabular Data in Your Terminal

**Tired of wrestling with spreadsheets and complex data analysis tools?** VisiData is a powerful terminal interface that lets you explore, manipulate, and understand tabular data directly from your command line.  [Check out the original repo](https://github.com/saulpw/visidata) to learn more.

[![Tests](https://github.com/saulpw/visidata/workflows/visidata-ci-build/badge.svg)](https://github.com/saulpw/visidata/actions/workflows/main.yml)
[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/saulpw/visidata)
[![discord](https://img.shields.io/discord/880915750007750737?label=discord)](https://visidata.org/chat)
[![mastodon @visidata@fosstodon.org][2.1]][2]
[![twitter @VisiData][1.1]][1]

![Frequency table](http://visidata.org/freq-move-row.gif)

## Key Features of VisiData

*   **Intuitive Terminal Interface:** Navigate and manipulate data with keyboard shortcuts, similar to Vim or Emacs.
*   **Wide Format Support:** Works with TSV, CSV, SQLite, JSON, XLSX (Excel), HDF5, and [many other formats](https://visidata.org/formats).
*   **Real-Time Data Exploration:** Quickly filter, sort, and summarize your data on the fly.
*   **Powerful Commands:** Hundreds of commands are available for data transformation, analysis, and more.
*   **Extensible with Plugins:** Customize VisiData to suit your specific needs.

## Getting Started

### Prerequisites

*   Linux, OS/X, or Windows (with WSL)
*   Python 3.8+
*   Additional Python modules may be needed for certain formats.

### Installation

Install VisiData using pip:

```bash
pip3 install visidata
```

Or, install the cutting-edge develop branch:

```bash
pip3 install git+https://github.com/saulpw/visidata.git@develop
```

For detailed installation instructions, including platform-specific guidance and package managers, see [visidata.org/install](https://visidata.org/install).

### Usage

Open a data file or pipe data into VisiData:

```bash
vd <input_file>
<command> | vd
```

Press `Ctrl+Q` to quit.

Explore the many available commands and options for data manipulation and analysis.

## Documentation and Resources

*   [VisiData documentation](https://visidata.org/docs)
*   [Plugin Author's Guide and API Reference](https://visidata.org/docs/api)
*   [Quick reference](https://visidata.org/man) (available within `vd` with `Ctrl+H`)
*   [Intro to VisiData Tutorial](https://jsvine.github.io/intro-to-visidata/)

## Get Help and Contribute

*   Report issues or ask questions: [Create an issue on Github](https://github.com/saulpw/visidata/issues).
*   Chat with the community:  #visidata on [irc.libera.chat](https://libera.chat/).
*   Support the project: [Support me on Patreon](https://www.patreon.com/saulpw)!

## License

VisiData is available under the GPLv3 License.

## Credits

*   Developed by: Saul Pwanson `<vd@saul.pw>`
*   Documentation and Packaging: Anja Kefala `<anja.kefala@gmail.com>`
*   Special thanks to all [contributors](https://visidata.org/credits/) and users.

[1.1]: http://i.imgur.com/tXSoThF.png
[1]: http://www.twitter.com/VisiData
[2.1]: https://raw.githubusercontent.com/mastodon/mastodon/main/app/javascript/images/logo.svg
[2]: https://fosstodon.org/@visidata