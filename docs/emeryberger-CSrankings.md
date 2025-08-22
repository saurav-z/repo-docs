# CSrankings: A Data-Driven Ranking of Top Computer Science Schools

CSrankings provides a **metrics-based ranking of top computer science schools**, offering a transparent and objective assessment of research productivity.

[View the original repository on GitHub](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Metrics-Based Ranking:**  Utilizes publication counts at top-tier computer science conferences, avoiding subjective survey-based methodologies.
*   **Objective & Transparent:**  Provides a data-driven approach to ranking, aiming to be difficult to manipulate and fostering trust.
*   **Focus on Research:**  Identifies institutions and faculty actively engaged in cutting-edge computer science research across various areas.
*   **Community Driven:** Relies on contributions to maintain the data.
*   **Easy to Contribute:** Offers a simplified contribution method via shallow cloning, promoting community involvement.

## How CSrankings Works:

The ranking methodology focuses on the number of publications by faculty at leading computer science conferences. This approach emphasizes research output and aims to create a more robust ranking system compared to survey-based methods.

## Contributing

Contributions to the CSrankings project are welcome! You can add or modify affiliations by editing the `csrankings-[a-z].csv` files.  Please review the [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

**Note:** Updates are processed quarterly.

## Getting Started Locally:

To run the site locally, you'll need to download the DBLP data and install the necessary dependencies.  Follow these steps:

1.  Run `make update-dblp` to download the DBLP data (requires ~19GB memory).
2.  Run `make` to rebuild the databases.
3.  Start a local web server (e.g., `python3 -m http.server`) and connect to [http://0.0.0.0:8000](http://0.0.0.0:8000).

**Required Dependencies:**

*   libxml2-utils
*   npm
*   typescript
*   google-closure-compiler
*   python-lxml
*   [pypy](https://doc.pypy.org/en/latest/install.html)
*   basex

Install dependencies via `apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler`

## Quick Contribution via Shallow Clone

For quick contributions, use the following steps:

1.  Fork the CSrankings repository.
2.  Clone your fork with `git clone --depth 1 https://github.com/yourusername/CSrankings`.
3.  Make changes on a branch, push them to your fork, and create a pull request.

## Acknowledgements

CSrankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). It builds upon the work of Swarat Chaudhuri and others, utilizing data from DBLP.org (under the ODC Attribution License) and a faculty dataset from Papoutsaki et al.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).