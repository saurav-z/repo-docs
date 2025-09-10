# All the Places: Extracting Point of Interest (POI) Data with Web Scraping

**All the Places is a powerful web scraping project designed to gather and standardize Point of Interest (POI) data from various websites.**  This project leverages the `scrapy` framework to extract POI information and provides a consistent data format.

➡️  **[Visit the original All the Places repository on GitHub](https://github.com/alltheplaces/alltheplaces)**

## Key Features:

*   **Web Scraping for POI Data:** Extracts location information from websites with store location pages.
*   **`scrapy` Framework:** Utilizes the popular Python-based `scrapy` web scraping framework for efficient data extraction.
*   **Standardized Data Format:**  Publishes results in a consistent and structured format.
*   **Modular Spiders:** Employs individual `scrapy` spiders for specific sites, facilitating maintainability and scalability.
*   **Weekly Updates:** The project is run weekly, and output is published to [alltheplaces.xyz](https://www.alltheplaces.xyz/).
*   **Open Source and Open Data:** The spider software is licensed under the MIT license, and the data generated is released under Creative Commons' CC-0 waiver.

## Getting Started

### Development Setup

Follow the instructions below to set up your development environment. Please note that Windows users may need to follow the instructions on the [scrapy docs](https://docs.scrapy.org/en/latest/intro/install.html#windows) for the most up-to-date information.

#### Ubuntu

These instructions were tested with Ubuntu 24.04 LTS on 2024-02-21.

1.  Install `uv`:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    ```

2.  Clone the repository:

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

3.  Install dependencies using `uv`:

    ```bash
    cd alltheplaces
    uv sync
    ```

4.  Verify installation:

    ```bash
    uv run scrapy
    ```

#### macOS

These instructions were tested with macOS 15.3.2 on 2025-04-01.

1.  Install `uv`:

    ```bash
    brew install uv
    ```

2.  Clone the repository:

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

3.  Install dependencies using `uv`:

    ```bash
    cd alltheplaces
    uv sync
    ```

4.  Verify installation:

    ```bash
    uv run scrapy
    ```

#### Codespaces

Develop in a cloud-based environment using GitHub Codespaces:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)

#### Docker

Utilize Docker for containerized development:

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

Contribute to the project by developing spiders and improving data extraction.  We provide guides to help you:

*   [Spider Naming](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps](docs/SITEMAP.md)
*   [Extracting data from structured data](docs/STRUCTURED_DATA.md)
*   [Pull Request guidelines](docs/PULL_REQUEST.md)
*   [Project Pipelines](docs/PIPELINES.md)

## Contact Us

For communication and contributions, use the GitHub [issue tracker](https://github.com/alltheplaces/alltheplaces/issues).  Contributors are also on the [OSM US Slack](https://slack.openstreetmap.us/) in the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel.

## License

*   **Data:**  Released under [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/) and published on [alltheplaces.xyz](https://alltheplaces.xyz/).
*   **Spider Software:** Licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).