# All the Places: Scrape Point of Interest (POI) Data from the Web

**All the Places** is a Python-based project designed to automatically extract and standardize Point of Interest (POI) data from websites, using web scraping techniques to build a comprehensive database of locations. (See original repo: [alltheplaces](https://github.com/alltheplaces/alltheplaces)).

## Key Features

*   **Automated POI Data Extraction:** Scrapes websites with store location pages to gather POI data.
*   **Python & Scrapy-Based:** Leverages the powerful Scrapy framework for efficient web scraping.
*   **Standardized Data Format:**  Publishes results in a consistent and easy-to-use format.
*   **Open Source & Collaborative:**  Contribute to the project and help improve the data collection.
*   **Weekly Data Updates:** The project's output is published regularly, ensuring fresh data (visit alltheplaces.xyz).

## Getting Started

### Development Setup

Follow these instructions to set up a development environment and start contributing:

#### Prerequisites

*   **Python:** Ensure you have Python installed on your system.
*   **Git:**  Make sure Git is installed for cloning the repository.
*   **uv:**  A package manager for fast dependency installation.  (Follow the instructions below, specific to your OS.)

#### Installation Steps (Ubuntu)

1.  Install `uv`:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    ```

2.  Clone the repository:

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    cd alltheplaces
    ```

3.  Install project dependencies:

    ```bash
    uv sync
    ```

4.  Test your installation:

    ```bash
    uv run scrapy
    ```

    If the above command runs without errors, your environment is set up correctly.

#### Installation Steps (macOS)

1.  Install `uv` (if you don't have it yet):

    ```bash
    brew install uv
    ```

2.  Clone the repository:

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    cd alltheplaces
    ```

3.  Install project dependencies:

    ```bash
    uv sync
    ```

4.  Test your installation:

    ```bash
    uv run scrapy
    ```

    If the above command runs without errors, your environment is set up correctly.

#### GitHub Codespaces

For cloud-based development, use GitHub Codespaces:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)

#### Docker

You can also use Docker for a containerized development environment:

1.  Clone the repository:

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    cd alltheplaces
    ```

2.  Build the Docker image:

    ```bash
    docker build -t alltheplaces .
    ```

3.  Run the Docker container:

    ```bash
    docker run --rm -it alltheplaces
    ```

### Contributing Code

Contribute to the project by creating spiders, improving existing ones, or adding features.  Review the following guides to help you contribute:

*   [Spider Naming](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps](docs/SITEMAP.md)
*   [Structured Data](docs/STRUCTURED_DATA.md)
*   [Pull Request Guidelines](docs/PULL_REQUEST.md)
*   [Behind the Scenes (Pipelines)](docs/PIPELINES.md)
*   [API Spider](docs/API_SPIDER.md)

### Weekly Runs and Data Publication

The project runs weekly, and its output is published on the [alltheplaces.xyz](https://www.alltheplaces.xyz/) website.

## Contact & Community

*   **Issue Tracker:** Use the GitHub [issue tracker](https://github.com/alltheplaces/alltheplaces/issues) for communication and reporting issues.
*   **OSM US Slack:**  Join the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel on OSM US Slack for discussions.

## License

*   **Data:**  Released under the [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/).
*   **Software (this repository):** Licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).