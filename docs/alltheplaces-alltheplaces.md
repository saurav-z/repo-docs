# All the Places: Your Source for Open Point of Interest (POI) Data

**All the Places is an open-source project dedicated to gathering and providing comprehensive point of interest (POI) data from various websites.** This project utilizes web scraping techniques to extract store location information and other valuable POI data. For the original source, see the [All the Places](https://github.com/alltheplaces/alltheplaces/) repository.

## Key Features:

*   **Web Scraping for POI Data:** Employs `scrapy`, a robust Python web scraping framework, to extract data.
*   **Standardized Data Format:** Publishes results in a consistent, easy-to-use format.
*   **Community Driven:** Encourages contributions through spider development and data refinement.
*   **Regular Data Updates:** Data is processed and published on a regular cadence to [alltheplaces.xyz](https://www.alltheplaces.xyz/).
*   **Open Data License:** Data generated is available under a [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/).
*   **MIT License:** The spider software is licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).

## Getting Started

### Development Setup

Follow these steps to set up your development environment:

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

3.  Install project dependencies:

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

2.  Clone the repository:

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

3.  Install project dependencies:

    ```bash
    cd alltheplaces
    uv sync
    ```

4.  Test the installation:

    ```bash
    uv run scrapy
    ```

#### Codespaces

You can use GitHub Codespaces for a cloud-based development environment:

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

Learn how to contribute spiders with these helpful guides:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

## Contact Us

*   **GitHub Issues:**  For communication, use the project's [issue tracker](https://github.com/alltheplaces/alltheplaces/issues).
*   **OSM US Slack:**  Find us on [OSM US Slack](https://slack.openstreetmap.us/) in the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel.