# All the Places: Scrape and Aggregate Point of Interest (POI) Data

**All the Places** is your go-to solution for collecting and standardizing point-of-interest (POI) data from the web, using the power of web scraping. ([Original Repo](https://github.com/alltheplaces/alltheplaces))

## Key Features

*   **Web Scraping for POI Data:** Extracts store location data from websites with location pages.
*   **Scrapy-Based Spiders:** Leverages the robust Scrapy framework (Python) to create and run individual spiders.
*   **Standardized Data Format:** Publishes results in a consistent format for easy data consumption and integration.
*   **Open-Source and Open Data:**  Contribute to the project and use the data for your own projects.
*   **Weekly Data Publication:**  Provides a regular stream of aggregated and cleaned POI data.

## Getting Started

This section provides instructions on how to set up a development environment to run and contribute to the project.

### Development Setup

Choose your preferred setup method:

#### Ubuntu

1.  **Install `uv`:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    ```

2.  **Clone the Repository:**

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

3.  **Install Dependencies using `uv`:**

    ```bash
    cd alltheplaces
    uv sync
    ```

4.  **Verify Installation:**

    ```bash
    uv run scrapy
    ```

    If this runs without errors, your setup is successful.

#### macOS

1.  **Install `uv` (using Homebrew):**

    ```bash
    brew install uv
    ```

2.  **Clone the Repository:**

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

3.  **Install Dependencies using `uv`:**

    ```bash
    cd alltheplaces
    uv sync
    ```

4.  **Verify Installation:**

    ```bash
    uv run scrapy
    ```

    Successful installation is indicated by a successful `scrapy` command execution.

#### Windows

1.  Follow the [Scrapy documentation](https://docs.scrapy.org/en/latest/intro/install.html#windows) for up-to-date installation instructions.
2.  **Clone the Repository:**

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

3.  **Install Dependencies using `uv`:**

    ```bash
    cd alltheplaces
    uv sync
    ```

4.  **Verify Installation:**

    ```bash
    uv run scrapy
    ```

    If this runs without errors, your setup is successful.

#### Codespaces

Develop in a cloud-based environment directly from GitHub:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)

#### Docker

1.  **Clone the Repository:**

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

2.  **Build the Docker Image:**

    ```bash
    cd alltheplaces
    docker build -t alltheplaces .
    ```

3.  **Run the Docker Container:**

    ```bash
    docker run --rm -it alltheplaces
    ```

### Contributing Code

We welcome contributions! Find guides on how to contribute:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

### Weekly Run

The aggregated POI data is published weekly to [alltheplaces.xyz](https://www.alltheplaces.xyz/).  For the best results, do not run all spiders at once.

## Contact Us

For questions, suggestions, and collaboration, please use the GitHub [issue tracker](https://github.com/alltheplaces/alltheplaces/issues). You can also find many contributors on the [OSM US Slack](https://slack.openstreetmap.us/), in the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel.

## License

*   **Data:** Released under the [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/).
*   **Spider Software (this repository):** Licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).