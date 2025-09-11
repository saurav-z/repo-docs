# All the Places: Scrape & Collect Point of Interest (POI) Data

**All the Places is a Python-based project leveraging web scraping to gather and standardize Point of Interest (POI) data from various websites.**

[Go to the original repository](https://github.com/alltheplaces/alltheplaces)

## Key Features

*   **Web Scraping Powerhouse:** Uses [Scrapy](https://scrapy.org/), a robust Python web scraping framework, to extract POI data.
*   **Standardized Data Format:**  Publishes results in a consistent, easy-to-use format ([DATA\_FORMAT.md](DATA_FORMAT.md)).
*   **Open Data:** Generated data is released under [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/).
*   **Community Driven:** Active project with guides for contributing code and a welcoming community.
*   **Regular Updates:** Weekly data runs and output published on [alltheplaces.xyz](https://www.alltheplaces.xyz/).

## Getting Started

### Development Setup

Follow these steps to set up your development environment:

**1. Install `uv` (Recommended for dependency management):**

   *   **Ubuntu:**
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.local/bin/env
        ```
   *   **macOS:**
        ```bash
        brew install uv
        ```
   *   **Windows:**  Follow the [Scrapy installation instructions](https://docs.scrapy.org/en/latest/intro/install.html#windows) if needed.

**2. Clone the Repository:**

   ```bash
   git clone git@github.com:alltheplaces/alltheplaces.git
   cd alltheplaces
   ```

**3. Install Dependencies:**

   ```bash
   uv sync  # Use 'uv sync' for efficient dependency management
   ```

**4. Verify Installation:**

   ```bash
   uv run scrapy
   ```

   If this command runs without errors, your installation is successful.

### Alternative Development Environments

*   **GitHub Codespaces:** Click the button below to launch a pre-configured cloud-based development environment:

    [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)

*   **Docker:**
    1.  Build the Docker image:
        ```bash
        docker build -t alltheplaces .
        ```
    2.  Run the Docker container:
        ```bash
        docker run --rm -it alltheplaces
        ```

## Contributing Code

We welcome contributions!  Here are resources to help you get started:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

## Contact Us

*   **GitHub Issues:**  Report bugs, suggest features, and discuss the project using the [issue tracker](https://github.com/alltheplaces/alltheplaces/issues).
*   **OSM US Slack:**  Join the conversation in the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel on [OSM US Slack](https://slack.openstreetmap.us/).

## License

*   **Data:**  Released under [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/).
*   **Code:**  Licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).