# All the Places: Extracting Point of Interest (POI) Data with Web Scraping

**All the Places** is a project dedicated to generating comprehensive point of interest (POI) data by scraping store location pages from various websites. Get the most up-to-date business location data with ease.

[Explore the project on GitHub](https://github.com/alltheplaces/alltheplaces)

## Key Features:

*   **POI Data Extraction:** Scrapes websites to gather POI data, including store locations, addresses, and contact information.
*   **Open-Source:** Uses a popular Python-based web scraping framework, Scrapy.
*   **Standardized Data Format:** Publishes results in a [standard format](DATA_FORMAT.md) for easy consumption and integration.
*   **Community Driven:** Contributions welcome!  Learn how to contribute to the project and develop your own spiders to expand data coverage.
*   **Weekly Runs & Data Publishing:** Data is regularly processed and published on the alltheplaces.xyz website.
*   **Flexible Deployment Options:** Supports development in multiple environments including Ubuntu, macOS, Codespaces, and Docker.
*   **Open License:** The generated data is available under [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/). The spider software is licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).

## Getting Started

### Development Setup

Follow these instructions to set up your development environment:

#### Ubuntu

1.  Install `uv`:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    ```

2.  Clone the project:

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

3.  Install dependencies:

    ```bash
    cd alltheplaces
    uv sync
    ```

4.  Test the installation:

    ```bash
    uv run scrapy
    ```

#### macOS

1.  Install `uv`:

    ```bash
    brew install uv
    ```

2.  Clone the project:

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

3.  Install dependencies:

    ```bash
    cd alltheplaces
    uv sync
    ```

4.  Test the installation:

    ```bash
    uv run scrapy
    ```

#### Codespaces

Use GitHub Codespaces for a cloud-based development environment:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)

#### Docker

1.  Clone the project:

    ```bash
    git clone git@github.com/alltheplaces/alltheplaces.git
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

Contribute to the project! Find useful guides for developing spiders:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

### The Weekly Run

The project's output is published on a regular cadence to the website: [alltheplaces.xyz](https://www.alltheplaces.xyz/).

## Contact Us

*   **Issue Tracker:** Submit issues and get help on the project's [issue tracker](https://github.com/alltheplaces/alltheplaces/issues).
*   **OSM US Slack:** Many contributors are active in the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel on OSM US Slack.