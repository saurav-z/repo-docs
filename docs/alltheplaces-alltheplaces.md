# All the Places: Your Source for Open Point of Interest (POI) Data

**Extract and standardize point of interest (POI) data from the web with All the Places, a powerful web scraping project.**  Find the original project on GitHub: [https://github.com/alltheplaces/alltheplaces](https://github.com/alltheplaces/alltheplaces)

## Key Features:

*   **Web Scraping for POI Data:** Scrapes websites to extract valuable POI data.
*   **Standardized Data Format:**  Publishes results in a consistent, easy-to-use format.
*   **Open Source & Accessible:**  The project and its data are open-source and free to use.
*   **Uses Scrapy:** Leverages the robust and popular Python web scraping framework, Scrapy.
*   **Regular Updates:** The project is regularly run to provide up-to-date POI data.
*   **Flexible Deployment:** Supports development setup on Ubuntu, macOS, Codespaces, and Docker.
*   **Community-Driven:**  Open to contributions with detailed guides and documentation.

## Getting Started

### Development Setup

Follow these steps to set up your development environment based on your preferred operating system:

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

Contribute to the project and help improve POI data coverage!  Refer to the following guides:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

## Weekly Runs & Data Publication

The project's output is published regularly on [alltheplaces.xyz](https://www.alltheplaces.xyz/), providing a continually updated source of POI data.

## Contact & Support

*   **GitHub Issues:**  Report issues and communicate through the project's [issue tracker](https://github.com/alltheplaces/alltheplaces/issues).
*   **OSM US Slack:**  Join the conversation in the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel on OSM US Slack.

## License

*   **Data:** Released under [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/).
*   **Spider Software:** Licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).