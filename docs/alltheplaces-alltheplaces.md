# All the Places: Scrape & Aggregate Point of Interest (POI) Data

**Unlock a comprehensive database of point-of-interest (POI) data by leveraging web scraping with All the Places.**  This project utilizes the powerful [Scrapy](https://scrapy.org/) framework to gather location information from websites, providing valuable POI data in a standardized format.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/alltheplaces/alltheplaces)

**Key Features:**

*   **Web Scraping for POI Data:** Automatically extracts location data from websites with store location pages.
*   **Scrapy Framework:** Built upon the robust and efficient Scrapy web scraping framework.
*   **Standardized Data Format:**  Publishes scraped data in a consistent format, streamlining integration.
*   **Weekly Data Updates:**  Benefit from regularly updated POI data, published on [alltheplaces.xyz](https://www.alltheplaces.xyz/).
*   **Open Source & Open Data:** The project is open-source (MIT License) and the data is released under Creative Commons’ CC-0 waiver.

## Getting Started

### Development Setup

Follow these steps to set up a development environment and start contributing to the project.

#### Ubuntu

Tested with Ubuntu 24.04 LTS (2024-02-21).

1.  **Install `uv`:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    ```
2.  **Clone the Repository:**

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```
3.  **Install Dependencies:**

    ```bash
    cd alltheplaces
    uv sync
    ```
4.  **Verify Installation:**

    ```bash
    uv run scrapy
    ```

    If no errors occur, your installation is functional.

#### macOS

Tested with macOS 15.3.2 (2025-04-01).

1.  **Install `uv`:**

    ```bash
    brew install uv
    ```
2.  **Clone the Repository:**

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```
3.  **Install Dependencies:**

    ```bash
    cd alltheplaces
    uv sync
    ```
4.  **Verify Installation:**

    ```bash
    uv run scrapy
    ```

    If no errors occur, your installation is functional.

#### Codespaces

Use GitHub Codespaces for a cloud-based development environment.  Click the button above to launch it.

#### Docker

Use Docker for containerized development:

1.  **Clone the Repository:**

    ```bash
    git clone git@github.com:alltheplaces/alltheplaces.git
    ```
2.  **Build the Docker Image:**

    ```bash
    cd alltheplaces
    docker build -t alltheplaces .
    ```
3.  **Run the Docker Container:**

    ```bash
    docker run --rm -it alltheplaces
    ```

### Contributing Code

We welcome contributions!  Learn how to contribute with these guides:

*   [What should I call my spider?](docs/SPIDER_NAMING.md)
*   [Using Wikidata and the Name Suggestion Index](docs/WIKIDATA.md)
*   [Sitemaps make finding POI pages easier](docs/SITEMAP.md)
*   [Data from many POI pages can be extracted without writing code](docs/STRUCTURED_DATA.md)
*   [What is expected in a pull request?](docs/PULL_REQUEST.md)
*   [What we do behind the scenes](docs/PIPELINES.md)

### Weekly Runs & Data Publication

The project runs spiders regularly, and the resulting data is published weekly on [alltheplaces.xyz](https://www.alltheplaces.xyz/).  Please be mindful of website usage when developing your own spiders.

## Contact Us

*   **GitHub Issues:** Use the [issue tracker](https://github.com/alltheplaces/alltheplaces/issues) for communication.
*   **OSM US Slack:** Find many contributors in the [#alltheplaces](https://osmus.slack.com/archives/C07EY4Y3M6F) channel on OSM US Slack.

## License

*   **Data:** Released under [Creative Commons’ CC-0 waiver](https://creativecommons.org/publicdomain/zero/1.0/).
*   **Spider Software (This Repository):** Licensed under the [MIT license](https://github.com/alltheplaces/alltheplaces/blob/master/LICENSE).

[Back to the GitHub Repository](https://github.com/alltheplaces/alltheplaces)
```
Key improvements and SEO optimizations:

*   **Clear, Concise Hook:** The one-sentence hook immediately grabs attention and describes the project's core function.
*   **Keyword-Rich Headings:** Uses relevant keywords like "POI," "Web Scraping," "Scrapy," and "Data Aggregation" in the headings.
*   **Bulleted Key Features:** Highlights the project's core benefits in an easy-to-scan format.
*   **SEO-Friendly Language:**  Uses terms that people would search for (e.g., "point of interest data," "scrape website data").
*   **Emphasis on Benefits:** Focuses on what the project *does* for the user, rather than just describing the code.
*   **Strong Call to Action:** Encourages use with easy setup instructions.
*   **Clear Contribution Guidance:** Provides direct links to contribution guidelines.
*   **Comprehensive Contact Information:** Makes it easy for users to get in touch and get support.
*   **Explicit Licensing:**  Clearly states the licensing for both the data and the code.
*   **Links Back to Repo:** Reinforces the call to action at the end of the document with a final link.