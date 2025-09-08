# All the Places: Extracting Point of Interest (POI) Data with Web Scraping

**All the Places is a powerful web scraping project designed to gather Point of Interest (POI) data from websites, providing a valuable resource for developers and data enthusiasts.**  Access the original project on GitHub: [https://github.com/alltheplaces/alltheplaces](https://github.com/alltheplaces/alltheplaces).

## Key Features:

*   **POI Data Extraction:** Scrapes websites with "store location" pages to extract comprehensive POI data.
*   **Scrapy Framework:** Utilizes the robust [Scrapy](https://scrapy.org/) web scraping framework for efficient data retrieval.
*   **Standardized Data Format:** Outputs data in a consistent format for easy integration and use.
*   **Modular Design:** Employs individual Scrapy spiders for each website, allowing for maintainability and scalability.
*   **Open Source & Open Data:**  Data generated is released under [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/) and the software is licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).

## Getting Started

### Development Setup

Follow the instructions below for setting up your development environment based on your operating system.

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

Easily set up a cloud-based development environment using GitHub Codespaces:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)

#### Docker

Use Docker for a containerized development environment:

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

Contribute to the project by creating spiders and helping to improve the data extraction process.  Please review the following guides:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

### Weekly Run & Data Publication

The project's output is published regularly on [alltheplaces.xyz](https://www.alltheplaces.xyz/).

## Contact Us

*   **Issue Tracker:**  For communication, please use the project's [GitHub issue tracker](https://github.com/alltheplaces/alltheplaces/issues).
*   **OSM US Slack:** Many contributors are present in the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel on [OSM US Slack](https://slack.openstreetmap.us/).