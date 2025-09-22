# All the Places: Your Source for Open Point of Interest (POI) Data

**All the Places** is a powerful project leveraging web scraping to generate a comprehensive database of Point of Interest (POI) data. [Explore the original repository](https://github.com/alltheplaces/alltheplaces/)!

## Key Features

*   **Automated POI Data Extraction:** Scrapes websites with store location pages to gather POI data.
*   **Scrapy-Powered:** Utilizes the popular Python-based Scrapy framework for efficient web scraping.
*   **Standardized Data Format:** Publishes results in a consistent format for easy integration.
*   **Open Data:** The data generated is released under Creative Commons’ CC-0 waiver.
*   **Open Source:** The spider software is licensed under the MIT license.

## Getting Started

This section provides instructions to get you up and running with All the Places.

### Development Setup

Choose your preferred setup method:

#### Ubuntu

1.  Install `uv`:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    ```

2.  Clone the repository:

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

3.  Install dependencies:

    ```bash
    cd alltheplaces
    uv sync
    ```

4.  Verify installation:

    ```bash
    uv run scrapy
    ```

#### macOS

1.  Install `uv`:

    ```bash
    brew install uv
    ```

2.  Clone the repository:

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

3.  Install dependencies:

    ```bash
    cd alltheplaces
    uv sync
    ```

4.  Verify installation:

    ```bash
    uv run scrapy
    ```

#### Codespaces

Develop in the cloud with GitHub Codespaces:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)

#### Docker

1.  Clone the repository:

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

2.  Build the Docker image:

    ```bash
    cd alltheplaces
    docker build -t alltheplaces .
    ```

3.  Run the Docker container:

    ```bash
    docker run --rm -it alltheplaces
    ```

### Contributing Code

Contribute to the project with these resources:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

### Weekly Run & Data Publication

The project's output is regularly published on our website at [alltheplaces.xyz](https://www.alltheplaces.xyz/).

## Contact

*   **Issue Tracker:** Report issues and discuss ideas via the [GitHub issue tracker](https://github.com/alltheplaces/alltheplaces/issues).
*   **OSM US Slack:** Connect with other contributors in the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel on OSM US Slack.

## License

*   **Data:** [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/)
*   **Software:** [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE)