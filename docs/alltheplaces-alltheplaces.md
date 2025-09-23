# All the Places: Scrape & Standardize Point of Interest (POI) Data

**All the Places is a powerful web scraping project designed to extract and standardize Point of Interest (POI) data from various websites, providing a valuable resource for developers and data enthusiasts.**

[Check out the original repository on GitHub](https://github.com/alltheplaces/alltheplaces)

## Key Features

*   **Automated POI Data Extraction:** Utilizes Scrapy, a robust Python web scraping framework, to automatically gather POI data.
*   **Data Standardization:**  Formats extracted data into a consistent, standard format for easy use and integration.
*   **Extensive Data Sources:** Scrapes data from websites with store location pages.
*   **Modular Design:** Uses individual "spider" scripts for each website, allowing for easy expansion and customization.
*   **Open Source & Community Driven:**  Contributions are welcome!

## Getting Started

This section provides instructions on setting up your development environment.

### Development Setup

Follow these steps to get the project running on your local machine:

#### Prerequisites

*   Python (version 3.7+)
*   Git

#### Ubuntu

1.  **Install `uv`:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    ```

2.  **Clone the repository:**

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

3.  **Navigate into the project directory:**

    ```bash
    cd alltheplaces
    ```

4.  **Install dependencies:**

    ```bash
    uv sync
    ```

5.  **Verify the installation:**

    ```bash
    uv run scrapy
    ```

#### macOS

1.  **Install `uv`:**

    ```bash
    brew install uv
    ```

2.  **Clone the repository:**

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

3.  **Navigate into the project directory:**

    ```bash
    cd alltheplaces
    ```

4.  **Install dependencies:**

    ```bash
    uv sync
    ```

5.  **Verify the installation:**

    ```bash
    uv run scrapy
    ```

#### Codespaces

You can quickly set up a cloud-based development environment using GitHub Codespaces:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)

#### Docker

Utilize Docker for a containerized development environment:

1.  **Clone the repository:**

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```

2.  **Build the Docker image:**

    ```bash
    cd alltheplaces
    docker build -t alltheplaces .
    ```

3.  **Run the Docker container:**

    ```bash
    docker run --rm -it alltheplaces
    ```

### Contributing Code

We welcome contributions!  Here are some resources to help you get started:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

### Weekly Run & Output

The project runs weekly, and the resulting data is published on the website: [alltheplaces.xyz](https://www.alltheplaces.xyz/).

## Contact Us

*   **GitHub Issues:** Report bugs, request features, or ask questions through the project's [issue tracker](https://github.com/alltheplaces/alltheplaces/issues).
*   **OSM US Slack:** Connect with contributors and get help in the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel on OSM US Slack.

## License

*   **Data:** Released under [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/) on [alltheplaces.xyz](https://www.alltheplaces.xyz/).
*   **Spider Software (Repository):** Licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).