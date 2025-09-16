# All the Places: Scrape and Aggregate Point of Interest (POI) Data

**All the Places** is an open-source project using web scraping to gather and structure Point of Interest (POI) data from diverse online sources. Check out the [original repo](https://github.com/alltheplaces/alltheplaces).

## Key Features

*   **Data Sourcing:** Scrapes POI data from websites with store location pages.
*   **Scrapy-Based:** Utilizes the powerful Python-based Scrapy framework for efficient web scraping.
*   **Standardized Output:** Formats scraped data into a consistent and accessible structure.
*   **Open Data:** Provides data under a Creative Commons CC-0 waiver, making it free to use.
*   **Open Source:** The spider software is licensed under the MIT license.
*   **Weekly Run:** Aggregated data is published weekly on [alltheplaces.xyz](https://www.alltheplaces.xyz/).

## Getting Started

Follow these instructions to set up a development environment.

### Development Setup

Choose your preferred environment: Ubuntu, macOS, GitHub Codespaces or Docker.

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

4.  Test installation:

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
    git clone git@github.com/alltheplaces/alltheplaces.git
    ```

3.  Install dependencies:

    ```bash
    cd alltheplaces
    uv sync
    ```

4.  Test installation:

    ```bash
    uv run scrapy
    ```

#### Codespaces

Use GitHub Codespaces for a cloud-based development environment:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)

#### Docker

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

We welcome contributions! Please refer to our guides to help you develop spiders:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

## Contact

*   **GitHub Issues:** Use the project's [issue tracker](https://github.com/alltheplaces/alltheplaces/issues).
*   **OSM US Slack:** Find us in the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel.

## License

*   **Data:** Released under [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/).
*   **Spider Software:** Licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).