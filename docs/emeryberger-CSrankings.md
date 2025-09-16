# CSrankings: The Metrics-Based Ranking of Top Computer Science Schools

**CSrankings provides a data-driven ranking of computer science institutions and faculty based on their research output at top-tier conferences.**  This approach offers a transparent and objective alternative to survey-based rankings, focusing on publication performance in selective venues.

[**View the original repository on GitHub**](https://github.com/emeryberger/CSrankings)

## Key Features

*   **Metrics-Driven:** Rankings based on the number of publications by faculty in leading computer science conferences.
*   **Transparent Methodology:** Avoids survey-based subjective evaluations, relying on measurable publication data.
*   **Difficult to Game:** Focuses on publications in highly selective conferences, making manipulation more challenging.
*   **Regular Updates:** Data is updated quarterly to reflect the latest research output.
*   **Community Driven:** Contributions are welcome for adding and updating faculty affiliations.

## Contributing

We welcome contributions to improve the accuracy and completeness of the CSrankings data.  Please review the [CONTRIBUTING.md](CONTRIBUTING.md) for detailed information on how to submit pull requests.

*   **Adding or Modifying Affiliations:**  All data is organized in CSV files. You can edit files directly in GitHub and submit pull requests.  Please note that updates are processed quarterly.

## Running CSrankings Locally

To run the website locally, you will need to:

1.  Download the DBLP data using `make update-dblp` (requires ~19GiB of memory).
2.  Rebuild the databases with `make`.
3.  Test by running a local web server (e.g., `python3 -m http.server`) and connecting to [http://0.0.0.0:8000](http://0.0.0.0:8000).

### Required Dependencies

You will need to install the following dependencies: `libxml2-utils`, `npm`, `typescript`, `closure-compiler`, `python-lxml`, [pypy](https://doc.pypy.org/en/latest/install.html), and basex.  For example:

```bash
apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
```

## Quick Contribution via a Shallow Clone

If you want to contribute without a full local clone:

1.  Fork the CSrankings repo.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`.
3.  Make your changes on a branch, push them to your clone, and create a pull request on GitHub.

## Acknowledgements

CSrankings was developed primarily by [Emery Berger](https://emeryberger.com), with contributions from a large community.  It builds upon the work of [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) (UT-Austin) and the dataset created by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html)

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).