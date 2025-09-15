# All the Places: Scrape Point of Interest (POI) Data with Python

**All the Places** is a powerful open-source project that leverages web scraping to extract valuable Point of Interest (POI) data from various websites, providing a comprehensive resource for location-based information. ([Original Repo](https://github.com/alltheplaces/alltheplaces))

## Key Features

*   **Web Scraping for POI Data:** Extracts location data from websites with store location pages.
*   **Scrapy Framework:** Utilizes the popular Python-based `scrapy` framework for efficient web scraping.
*   **Standard Data Format:** Publishes results in a standardized format for easy use and integration.
*   **Extensible Spiders:** Supports individual "spiders" for different websites, allowing for customization and scalability.
*   **Open Source & Community Driven:** Contribute to the project and help expand the data set.
*   **Regular Data Updates:** Data is regularly updated and published on [alltheplaces.xyz](https://www.alltheplaces.xyz/).

## Getting Started

### Development Setup

Follow these steps to set up your development environment:

**Prerequisites:**

*   Python 3.7+
*   Git

**Installation using `uv` (Recommended - Ubuntu & macOS):**

1.  **Install `uv`:**

    *   **Ubuntu:**
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.local/bin/env
        ```
    *   **macOS:**
        ```bash
        brew install uv
        ```
2.  **Clone the repository:**

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    cd alltheplaces
    ```
3.  **Install dependencies:**

    ```bash
    uv sync
    ```
4.  **Verify Installation:**

    ```bash
    uv run scrapy
    ```

**Windows Installation:**
*   Follow the [scrapy docs](https://docs.scrapy.org/en/latest/intro/install.html#windows) for Windows installation.

**Using GitHub Codespaces:**

Click the button below to launch a pre-configured development environment in the cloud:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)

**Using Docker:**

1.  **Clone the repository:**

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    cd alltheplaces
    ```
2.  **Build the Docker image:**

    ```bash
    docker build -t alltheplaces .
    ```
3.  **Run the Docker container:**

    ```bash
    docker run --rm -it alltheplaces
    ```

### Contributing Code

We welcome contributions!  Refer to the following guides for developing spiders and contributing to the project:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

### The Weekly Run

The project's output is published weekly on [alltheplaces.xyz](https://www.alltheplaces.xyz/).

## Contact

For communication, use the project's GitHub [issue tracker](https://github.com/alltheplaces/alltheplaces/issues).  Contributors are also active on the [OSM US Slack](https://slack.openstreetmap.us/) in the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel.

## License

*   **Data:** Released under [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/).
*   **Spider Software:** Licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).