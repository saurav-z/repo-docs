# CSrankings: The Premier Metrics-Based Ranking of Computer Science Schools

**Looking to assess top computer science programs?** CSrankings provides a data-driven, metrics-based ranking of computer science schools, relying on publication data from highly selective conferences to offer a transparent and objective view of research excellence.  This approach avoids the subjective bias often found in survey-based rankings.  Check out the [original repo](https://github.com/emeryberger/CSrankings) for more details.

## Key Features:

*   **Metrics-Driven:**  Ranks schools based on faculty publications in the most selective computer science conferences, providing a quantitative and objective assessment.
*   **Difficult to Game:** The ranking methodology focuses on publications in top conferences, making it harder to manipulate compared to citation-based metrics.
*   **Comprehensive Data:**  Leverages data from DBLP.org to ensure a wide and up-to-date view of research output.
*   **Community-Driven:**  Welcomes contributions to improve the accuracy and coverage of faculty affiliations and other data (contributions are processed quarterly).
*   **Transparent Methodology:**  The ranking's methodology is open and accessible, with an FAQ available at [https://csrankings.org/faq.html](https://csrankings.org/faq.html).

## Contributing to CSrankings

CSrankings thrives on community contributions.  Help us keep the data accurate and up-to-date!

### Adding or Modifying Affiliations

1.  **Data Files:**  Edit the `csrankings-[a-z].csv` files, organizing data by the first letter of authors' first names.
2.  **Contribution Guidelines:**  Please read the `CONTRIBUTING.md` file in the repository for full details and instructions on how to contribute.
3.  **Quarterly Updates:**  Note that updates are processed on a quarterly basis.  Your pull requests are welcome at any time.

### Getting Started Locally

To run CSrankings locally:

1.  **DBLP Data:**  Run `make update-dblp` to download the DBLP data. (Requires ~19GB memory).
2.  **Build Databases:**  Run `make`.
3.  **Local Web Server:**  Start a local web server (e.g., `python3 -m http.server`).
4.  **Access:**  View the site at `http://0.0.0.0:8000`.

### Prerequisites

You will need to install the following:

*   `libxml2-utils` (or equivalent package with `xmllint`)
*   `npm`
*   `typescript`
*   `google-closure-compiler`
*   `python-lxml`
*   `pypy`
*   `basex`

Install these with:
`apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler`

### Quick Contribution via a Shallow Clone

To contribute changes without a full clone:

1.  Fork the CSrankings repo.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`.
3.  Make your changes on a branch, push to your fork, and create a pull request.

## Acknowledgements

CSrankings was developed primarily by and is maintained by [Emery Berger](https://emeryberger.com).  It incorporates feedback from numerous contributors.  The project builds upon the work of [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) and the faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).  Data from [DBLP.org](http://dblp.org) is used, licensed under the ODC Attribution License.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).