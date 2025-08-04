# CS Rankings: Your Guide to Top Computer Science Schools

**Looking for the best computer science programs?** CS Rankings provides a data-driven, metrics-based ranking of top computer science schools, focusing on faculty research output.

**Learn more and explore the rankings at [csrankings.org](https://csrankings.org/) or visit the original repository on GitHub: [emeryberger/CSrankings](https://github.com/emeryberger/CSrankings).**

## Key Features

*   **Metrics-Based Ranking:**  Ranks schools based on the number of publications by faculty in top computer science conferences, offering a data-driven alternative to survey-based approaches.
*   **Difficult to Game:** Designed to be resistant to manipulation compared to citation-based rankings.
*   **Comprehensive Data:** Leverages data from DBLP.org and community contributions.
*   **Community-Driven:**  Actively maintained and improved by contributors who add and update faculty affiliations and other data.
*   **Quarterly Updates:** Rankings are updated quarterly to ensure data freshness.

## How it Works

CS Rankings uses a metrics-based approach to rank computer science departments, focusing on the research output of faculty. The ranking is based on the number of publications by faculty in the most selective computer science conferences.

## Contributing

*   **Contribution Process:**  Contributions are processed quarterly; submit pull requests at any time.
*   **Data Files:**  Faculty and affiliation data are primarily stored in `csrankings-[a-z].csv` files.
*   **Contribution Guide:** See [`CONTRIBUTING.md`](CONTRIBUTING.md) (in the original repository) for detailed instructions on how to contribute.
*   **Quick Contribution:**  Use shallow cloning for faster contribution without a full repository download.

## Setting Up Locally

To run the website locally, follow these steps:

1.  Download DBLP data: `make update-dblp` (requires ~19GiB of memory).
2.  Rebuild the databases: `make`.
3.  Run a local web server:  `python3 -m http.server` and connect to `http://0.0.0.0:8000`.

**Dependencies:** You will need to install various dependencies, including `libxml2-utils`, `npm`, `typescript`, `closure-compiler`, `python-lxml`, `pypy`, and `basex`. Instructions are in the original README.

## Acknowledgements

The CS Rankings site was developed primarily by [Emery Berger](https://emeryberger.com) and built upon prior work by Swarat Chaudhuri and others. It also incorporates data from DBLP.org.

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).