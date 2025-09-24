# All the Places: Your Source for Open Point of Interest (POI) Data

**All the Places uses web scraping to gather comprehensive point of interest (POI) data, providing valuable information for developers, researchers, and map enthusiasts.** Check out the [original repo](https://github.com/alltheplaces/alltheplaces/) to get started.

## Key Features

*   **Data Acquisition:** Scrapes location data from websites with store location pages.
*   **Open Data:** Publishes the results in a standard format under a CC-0 license.
*   **Web Scraping Framework:** Utilizes `scrapy`, a powerful Python-based web scraping framework.
*   **Modular Design:** Employs individual site spiders for targeted data retrieval.
*   **Weekly Updates:** Publishes data on a regular cadence, keeping information current.

## Getting Started

### Development Setup

Follow these instructions to set up your development environment. Instructions are available for:

*   **Ubuntu**
*   **macOS**
*   **GitHub Codespaces:** Leverage cloud-based development environments.
*   **Docker:** Use containerized environments.

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

#### Codespaces

Click the button below to open the project in GitHub Codespaces:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)

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

### Contributing Code

We welcome contributions!  See the following guides for more information on developing spiders:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

### Weekly Run & Data Output

The project's output is published regularly on [alltheplaces.xyz](https://www.alltheplaces.xyz/).  Please avoid running all spiders to maintain good relationships with data sources.

## Contact

*   **Issue Tracker:** Report issues and discuss features on the project's GitHub [issue tracker](https://github.com/alltheplaces/alltheplaces/issues).
*   **OSM US Slack:** Join the conversation in the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel on OSM US Slack.

## License

*   **Data:** Released under [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/).
*   **Spider Software:** Licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).