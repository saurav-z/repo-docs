# CSrankings: Data-Driven Computer Science School Rankings

**CSrankings.org** provides a unique, metrics-based ranking of top computer science schools, offering a data-driven alternative to survey-based approaches. Explore the code and data behind this innovative ranking system on [GitHub](https://github.com/emeryberger/CSrankings).

## Key Features

*   **Metrics-Based Approach:**  Ranks institutions based on the number of publications by faculty in top-tier computer science conferences.
*   **Difficult-to-Game Methodology:** Uses conference publications as a primary metric to avoid manipulation common in citation-based rankings.
*   **Comprehensive Data:**  Utilizes data from DBLP and other sources to provide a broad view of research activity.
*   **Open Source:**  The code and data are available for anyone to view and contribute to the rankings.
*   **Community Driven:** Contributions are welcomed.

## Contributing

Contributions to improve the dataset are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.
*   **Data Files:** The core data is stored in `csrankings-[a-z].csv` files.
*   **Update Frequency:** Updates are processed quarterly.

## Running Locally

To run CSrankings locally, you'll need to:

1.  Download the DBLP data: `make update-dblp` (requires ~19GiB memory).
2.  Build the databases: `make`.
3.  Run a local web server (e.g., `python3 -m http.server`) and access it at `http://0.0.0.0:8000`.

### Prerequisites

Ensure you have the following installed:

*   libxml2-utils
*   npm
*   typescript
*   google-closure-compiler
*   python-lxml
*   pypy
*   basex

Install them with a command like:

``apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler``

## Quick Contribution via Shallow Clone

For easy contribution without a full clone:

1.  Fork the CSrankings repository.
2.  Shallow clone your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`.
3.  Make changes, push to your clone, and create a pull request.

## Acknowledgements

This project was developed by [Emery Berger](https://emeryberger.com) and is based on the work of [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/), [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html) and uses information from [DBLP.org](http://dblp.org).

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).