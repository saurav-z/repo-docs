# CSrankings: The Data-Driven Ranking of Top Computer Science Schools

**Tired of ranking methods based solely on surveys? CSrankings offers a metrics-based ranking of computer science schools, providing a transparent and data-driven approach to evaluating research excellence.**

[View the CSrankings website](https://csrankings.org/) | [View the Original Repository on GitHub](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Metrics-Based Approach:** Rankings are based on the number of publications by faculty in top computer science conferences, offering a transparent and objective assessment.
*   **Avoids Manipulation:** Unlike citation-based metrics, this approach aims to be difficult to manipulate, ensuring a more reliable ranking.
*   **Community Driven:**  Contribute to the rankings by adding or modifying affiliations.  Updates are processed quarterly.
*   **Open Source:**  The codebase and data are available for review, modification, and community contribution.
*   **Transparent Methodology:** Detailed methodology is available in the [FAQ](https://csrankings.org/faq.html)

## Contributing

### Adding or Modifying Affiliations

1.  **Data Format:**  Affiliations are organized in `csrankings-[a-z].csv` files, with authors alphabetized by first name.
2.  **Contribution Guidelines:**  Refer to `CONTRIBUTING.md` for detailed instructions.
3.  **Quarterly Updates:**  Pull requests are welcome anytime and are processed quarterly.

### Quick Contribution via Shallow Clone

For making changes without a full clone of the repository:

1.  Fork the CSrankings repo.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make your changes on a branch, push them to your clone, and create a pull request.
4.  For subsequent contributions, repeat the fork/shallow clone process to ensure an up-to-date base.

## Getting Started Locally

To run the site locally, you'll need to:

1.  Download DBLP data: `make update-dblp` (requires ~19GB of memory).
2.  Rebuild the databases: `make`
3.  Run a local web server (e.g., `python3 -m http.server`) and view the site at [http://0.0.0.0:8000](http://0.0.0.0:8000)

**Required Dependencies:**

*   libxml2-utils (or equivalent)
*   npm
*   typescript
*   google-closure-compiler
*   python-lxml
*   pypy
*   basex

Install these with a command like: `apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler`

## Acknowledgements

CSrankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). It is based on code and data collected by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) and the original faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

This site uses information from [DBLP.org](http://dblp.org) which is made available under the ODC Attribution License.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).