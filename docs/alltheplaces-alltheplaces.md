# All the Places: Scrape and Aggregate Point of Interest (POI) Data

**All the Places is an open-source project that uses web scraping to gather and standardize Point of Interest (POI) data from across the web.**

This project leverages the power of [Scrapy](https://scrapy.org/), a robust Python web scraping framework, to extract valuable location information from various websites. The scraped data is then published in a standardized format.

[Visit the original repository on GitHub](https://github.com/alltheplaces/alltheplaces).

## Key Features:

*   **Web Scraping for POI Data:** Extracts data from websites with "store location" pages.
*   **Scrapy-Powered Spiders:** Uses custom Scrapy spiders to retrieve POI data efficiently.
*   **Standardized Data Format:** Publishes results in a consistent, easy-to-use format.
*   **Regular Data Updates:** The project runs weekly, updating the data and making it available on [alltheplaces.xyz](https://www.alltheplaces.xyz/).
*   **Open Source & Open Data:** The spider software is licensed under the MIT License, and the scraped data is released under the Creative Commons CC-0 waiver.

## Getting Started

### Development Setup

Follow the instructions below to set up your development environment.

#### Prerequisites
*   **Python:** Ensure you have Python installed. It is recommended to use `uv` for dependency management.

#### Installation for Various Operating Systems:

##### Ubuntu

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

##### macOS

1.  Install `uv` using Homebrew:

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

##### Codespaces

You can set up a cloud-based development environment using GitHub Codespaces. Click the button to create a Codespace:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)

##### Docker

You can use Docker for a containerized development environment.

1.  Clone the project:

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

## Contributing Code

We welcome contributions! Here are some guides to assist you:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

## Contact Us

For communication, please use the GitHub [issue tracker](https://github.com/alltheplaces/alltheplaces/issues). You can also find contributors on the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel within the [OSM US Slack](https://slack.openstreetmap.us/).

## License

*   **Data:** Released under [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/).
*   **Spider Software:** Licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).