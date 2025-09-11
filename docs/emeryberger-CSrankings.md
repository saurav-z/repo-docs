# CS Rankings: Your Go-To Resource for Data-Driven Computer Science School Rankings

This project offers a unique, metrics-based approach to ranking top computer science schools and faculty, focusing on research output.  For more details, visit the [original repository on GitHub](https://github.com/emeryberger/CSrankings).

## Key Features:

*   **Metrics-Driven Ranking:** Rankings are based on the number of publications by faculty at highly selective computer science conferences.
*   **Objective Assessment:**  Avoids the limitations of survey-based rankings by using objective, quantifiable metrics.
*   **Transparent Methodology:**  Provides clear insights into the factors influencing the rankings, emphasizing research impact.
*   **Difficult-to-Game Approach:** Designed to be resistant to manipulation, focusing on publications in top-tier conferences.
*   **Community-Driven Data:**  Contributions are welcome and essential for keeping the data up-to-date.

## Contributing

### Adding or Modifying Affiliations

**_NOTE: Updates are now processed on a quarterly basis._**

All data is in the files `csrankings-[a-z].csv`. See `CONTRIBUTING.md` for details on how to contribute and the contribution process.

## Running Locally

To run this site, you'll need to download the DBLP data using `make update-dblp` (requires 19GiB of memory). Then, build the databases by running `make`. Test the site by running a local web server (e.g., `python3 -m http.server`) and connecting to [http://0.0.0.0:8000](http://0.0.0.0:8000).

### Dependencies

You will need to install:
*   `libxml2-utils`
*   `npm`
*   `typescript`
*   `google-closure-compiler`
*   `python-lxml`
*   `pypy`
*   `basex`

via a command line like:

``apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler``

## Quick Contribution via a Shallow Clone

1.  Fork the CSrankings repo.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make your changes on a branch, push them to your clone, and create a pull request on GitHub.

## Acknowledgements

Developed and maintained by [Emery Berger](https://emeryberger.com). The project incorporates extensive feedback and contributions from many individuals. The site uses information from [DBLP.org](http://dblp.org) which is made available under the ODC Attribution License.

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).