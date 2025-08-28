# CSrankings: The Metrics-Driven Ranking of Top Computer Science Schools

**CSrankings provides a data-driven, objective ranking of computer science institutions and faculty based on their research publications at top conferences.**  For more details, please visit the original repository: [https://github.com/emeryberger/CSrankings](https://github.com/emeryberger/CSrankings).

## Key Features:

*   **Metrics-Based Ranking:** Utilizes a metrics-driven approach, relying on publication counts in top computer science conferences, avoiding subjective survey-based methods.
*   **Objective & Difficult to Game:** Designed to be difficult to manipulate, unlike citation-based metrics.
*   **Comprehensive Data:** Uses a curated dataset and is constantly updated.
*   **Community-Driven:** Open-source and welcomes contributions from the community to improve data accuracy and maintain faculty affiliations.
*   **Transparent Methodology:** Provides detailed information on the ranking methodology and data sources (DBLP).

## Contributing to CSrankings

### Adding or Modifying Affiliations

*   **Quarterly Updates:** Updates are processed quarterly.
*   **Direct GitHub Editing:** You can submit pull requests by directly editing the `csrankings-[a-z].csv` files within the repository.
*   **Contribution Guidelines:** Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for full details on how to contribute.

### Running CSrankings Locally

1.  **Download DBLP Data:** Run `make update-dblp` (requires significant memory - approx. 19GiB).
2.  **Build Databases:** Run `make`.
3.  **Test:** Run a local web server (e.g., `python3 -m http.server`) and access it at [http://0.0.0.0:8000](http://0.0.0.0:8000).
4.  **Dependencies:** Install required dependencies: `libxml2-utils`, `npm`, `typescript`, `google-closure-compiler`, `python-lxml`, `pypy`, and `basex` via command line (see original README).

### Quick Contribution via Shallow Clone

1.  Fork the CSrankings repository.
2.  Shallow clone your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`.
3.  Make your changes on a branch, push them, and create a pull request on GitHub.

## Acknowledgements

CSrankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). It's built upon the work of Swarat Chaudhuri, the initial faculty affiliation dataset by Papoutsaki et al., and utilizes data from [DBLP.org](http://dblp.org) made available under the ODC Attribution License.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).