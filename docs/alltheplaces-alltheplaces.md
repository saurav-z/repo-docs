# All the Places: Scrape & Gather Point of Interest (POI) Data

**All the Places is an open-source project that leverages web scraping to collect and standardize point-of-interest (POI) data from various websites.**

[View the project on GitHub](https://github.com/alltheplaces/alltheplaces/)

## Key Features

*   **Automated POI Data Extraction:** Scrapes websites with store location pages to extract valuable POI data.
*   **Standardized Data Format:**  Outputs data in a consistent format for easy integration and use.
*   **Built on Scrapy:** Utilizes the powerful and flexible Scrapy web scraping framework, a popular Python-based framework.
*   **Open-Source & Community-Driven:**  Contribute to the project by writing spiders and improving data collection.
*   **Weekly Data Updates:**  Provides regularly updated POI data, accessible through the [alltheplaces.xyz](https://www.alltheplaces.xyz/) website.

## Getting Started

Follow these steps to set up a development environment:

### Development Setup

Choose your preferred method:

*   **Ubuntu:** Follow the instructions below.
*   **macOS:** Follow the instructions below.
*   **GitHub Codespaces:** Use the pre-configured cloud-based environment.
    [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)
*   **Docker:** Use the Docker instructions below.

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

## Contributing Code

We welcome contributions!  Check out these guides to help you develop spiders:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

## The Weekly Run

The project's output is regularly published on [alltheplaces.xyz](https://www.alltheplaces.xyz/).

## Contact

*   **GitHub Issues:**  Use the [issue tracker](https://github.com/alltheplaces/alltheplaces/issues) for communication.
*   **OSM US Slack:** Join the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel on OSM US Slack.

## License

*   **Data:** Released under [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/).
*   **Spider Software:** Licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).