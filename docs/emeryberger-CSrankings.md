# CSrankings: The Data-Driven Computer Science School Ranking

Tired of rankings based on surveys? CSrankings provides a **metrics-based ranking of top computer science schools** designed to identify institutions and faculty actively engaged in research across various areas of computer science.  See the original repository on GitHub: [emeryberger/CSrankings](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Metrics-Based Approach:** Ranks schools based on the number of publications by faculty in top computer science conferences, avoiding survey-based methodologies.
*   **Difficult-to-Game System:** Focuses on publications in highly selective conferences to create a ranking that is harder to manipulate compared to citation-based metrics.
*   **Regular Updates:** Data is updated on a quarterly basis.
*   **Community Contribution:** Contributions are welcome!  Help improve the accuracy and scope of the rankings by adding or modifying affiliations.
*   **Open Source:** All code and data are available for review and use.
*   **Uses DBLP:**  Leverages the DBLP database for comprehensive publication data.

## Contributing

**Contribution Guidelines:**  Contributions are currently processed on a quarterly basis. You can submit pull requests at any time. Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for full details on how to contribute.

**How to Contribute:**

*   Edit files directly in GitHub.  All data is in `csrankings-[a-z].csv` files.
*   Files are organized by the initial letter of the author's first name.
*   Follow the instructions in `CONTRIBUTING.md`.

## Running Locally

To run the site locally, you will need to:

1.  Download the DBLP data by running ``make update-dblp`` (requires ~19GiB of memory).
2.  Rebuild the databases by running ``make``.
3.  Run a local web server (e.g., ``python3 -m http.server``).
4.  Connect to [http://0.0.0.0:8000](http://0.0.0.0:8000).

**Dependencies:**

You will also need to install the following dependencies:

*   `libxml2-utils` (or equivalent)
*   `npm`
*   `typescript`
*   `google-closure-compiler`
*   `python-lxml`
*   `pypy`
*   `basex`

Install these dependencies using a command like:
``apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler``

## Quick Contribution via a Shallow Clone

To contribute without a full clone:

1.  Fork the CSrankings repo.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes on a branch, push them to your clone, and create a pull request.

## Acknowledgements

This site was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). It incorporates extensive feedback from many contributors. This site was initially based on code and data collected by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) (UT-Austin). The original faculty affiliation dataset was constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

This site uses information from [DBLP.org](http://dblp.org) which is made available under the ODC Attribution License.

## License

CSRankings is covered by the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).