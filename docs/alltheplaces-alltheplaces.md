# All the Places: Extract Point of Interest (POI) Data with Web Scraping

**Uncover a wealth of Point of Interest (POI) data from websites using this powerful web scraping project.**

[View the project on GitHub](https://github.com/alltheplaces/alltheplaces)

This project leverages the robust [Scrapy](https://scrapy.org/) framework to extract and standardize POI data from websites that feature store locations. Spiders are used to retrieve POI data, publishing the results in a [standard format](DATA_FORMAT.md).

## Key Features

*   **Web Scraping:** Utilizes Scrapy, a leading Python web scraping framework.
*   **POI Data Extraction:** Focuses on extracting data from websites with store location pages.
*   **Standardized Data:** Publishes results in a consistent, well-defined format.
*   **Open Source:** The project is open source and actively welcomes contributions.

## Getting Started

### Development Setup

Follow these steps to set up your development environment.

#### Prerequisites

*   Python 3.7+
*   Git

#### Installation (Choose your preferred method)

**Windows users:** Please follow the [Scrapy documentation](https://docs.scrapy.org/en/latest/intro/install.html#windows) for specific Windows installation steps.

**Ubuntu**

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

**macOS**

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

**GitHub Codespaces**

Click the button below to launch a pre-configured development environment in Codespaces:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)

**Docker**

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

### Contributing

We welcome contributions!  Please refer to the following guides for more information on contributing:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

### Weekly Run

The project's output is published weekly on our website, [alltheplaces.xyz](https://www.alltheplaces.xyz/).

## Contact

For communication, please use the project's GitHub [issue tracker](https://github.com/alltheplaces/alltheplaces/issues). Contributors are also often present on the [OSM US Slack](https://slack.openstreetmap.us/) in the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel.

## License

*   **Data:** Released under the [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/) on [alltheplaces.xyz](https://www.alltheplaces.xyz/).
*   **Spider Software (this repository):** Licensed under the [MIT License](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).