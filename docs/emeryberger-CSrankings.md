# CSrankings: Metrics-Based Ranking of Top Computer Science Schools

**CSrankings is a data-driven, metrics-based ranking system that assesses computer science institutions and faculty performance based on publications in top-tier conferences.** This approach provides a more objective evaluation compared to survey-based rankings. Explore the code and data behind this valuable resource on [GitHub](https://github.com/emeryberger/CSrankings).

## Key Features

*   **Metrics-Driven Approach:** Rankings are based on the number of publications in highly selective computer science conferences, providing a more objective assessment.
*   **Avoids Manipulation:**  Designed to be difficult to "game," unlike citation-based metrics that are often easier to manipulate.
*   **Open Source:** The code and data are publicly available, allowing for transparency and community contributions.
*   **Regular Updates:**  Data is updated on a quarterly basis to reflect the latest research output.
*   **Community Driven:** Encourages community contributions for adding and modifying faculty affiliations and other data.
*   **Based on DBLP Data:** Leverages the comprehensive DBLP database for publication information.

## Contributing

Contributions are welcome! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines on how to contribute.  Changes are processed quarterly.  You can edit files directly in GitHub.  All data is in the files `csrankings-[a-z].csv`, with authors listed in alphabetical order by their first name, organized by the initial letter.

### Quick Contribution via a Shallow Clone

1.  Fork the CSrankings repo.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make your changes on a branch, push them to your clone, and create a pull request on GitHub as usual.

## Running CSrankings Locally

To run the site locally, you will need to download the DBLP data and install the necessary dependencies.

1.  Run `make update-dblp` (requires ~19GB of memory).
2.  Run `make` to rebuild the databases.
3.  Test by running a local web server (e.g., `python3 -m http.server`) and connecting to [http://0.0.0.0:8000](http://0.0.0.0:8000).

**Dependencies:** libxml2-utils, npm, typescript, closure-compiler, python-lxml, pypy, and basex.  Install these using:
``apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler``

## Acknowledgements

Developed primarily by [Emery Berger](https://emeryberger.com). This project builds upon work by Swarat Chaudhuri, Papoutsaki et al., and others. Uses data from [DBLP.org](http://dblp.org) under the ODC Attribution License.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).