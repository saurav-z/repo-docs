# CSrankings: Your Data-Driven Guide to Top Computer Science Schools

**CSrankings provides a rigorous, metrics-based ranking of computer science institutions and faculty, offering a transparent and objective view of research performance.**  For more details and the live website, visit [csrankings.org](https://csrankings.org).

[![CSrankings Logo](https://csrankings.org/img/logo.png)](https://csrankings.org/)

CSrankings.org is a valuable resource for prospective students, faculty, and anyone interested in understanding the landscape of computer science research. Unlike rankings based on subjective surveys, CSrankings relies on a transparent, data-driven approach using publication data from leading computer science conferences.  This repository contains the code and data used to generate the rankings.  Explore the source code and contribute to the project!

## Key Features:

*   **Metrics-Based:** Rankings are determined by the number of publications by faculty in top-tier computer science conferences.
*   **Objective:** Avoids the subjectivity of survey-based rankings, providing a more reliable measure of research productivity.
*   **Transparent:** Utilizes openly available data, making the methodology and results easy to understand and verify.
*   **Easy to Contribute:** Contribute to the project by editing CSV files or forking the repo.
*   **Data-Rich:** Includes data from DBLP and other sources.

## Contributing

Want to improve the accuracy or completeness of CSrankings?  Contributions are welcome!

*   **Data Files:**  Affiliations and other data are stored in `csrankings-[a-z].csv` files.
*   **Contribution Guidelines:**  See `CONTRIBUTING.md` for detailed instructions on how to contribute.
*   **Quarterly Updates:**  Updates are processed on a quarterly basis.

### Quick Contribution via a Shallow Clone

1.  Fork the CSrankings repo.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make your changes on a branch, push them to your clone, and create a pull request on GitHub.

## Building Locally

To run the website locally, you'll need to:

1.  Download the DBLP data using `make update-dblp`. (Requires ~19GB of memory)
2.  Rebuild the databases with `make`.
3.  Run a local web server (e.g., `python3 -m http.server`) and connect to [http://0.0.0.0:8000](http://0.0.0.0:8000).
4.  Install dependencies:  `apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler`
5.  Install additional packages like: [pypy](https://doc.pypy.org/en/latest/install.html).

## Acknowledgements

This project was developed primarily by [Emery Berger](https://emeryberger.com). It builds upon the work of Swarat Chaudhuri (UT-Austin) and a faculty affiliation dataset constructed by Papoutsaki et al.

This site uses information from [DBLP.org](http://dblp.org) which is made available under the ODC Attribution License.

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

[Visit the original repository on GitHub](https://github.com/emeryberger/CSrankings)