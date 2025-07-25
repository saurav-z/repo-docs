# CS Rankings: A Data-Driven Ranking of Top Computer Science Schools

**CS Rankings** provides a comprehensive, metrics-based ranking of computer science departments worldwide, offering an objective assessment of research performance.  For the original source code and data, visit the [GitHub repository](https://github.com/emeryberger/CSrankings).

## Key Features:

*   **Metrics-Based Approach:**  Relies on the number of publications by faculty in top-tier computer science conferences, providing a data-driven and objective ranking.
*   **Difficult to Game:** Designed to be resistant to manipulation, unlike rankings based on surveys or easily manipulated citation metrics.
*   **Extensive Data Source:** Utilizes data from DBLP (Digital Bibliography & Library Project), ensuring a broad and up-to-date dataset.
*   **Community Driven:**  Allows contributions for adding or modifying affiliations.  Updates are processed quarterly.
*   **Open Source:**  The code and data are available under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

## Contributing

### Adding or Modifying Affiliations

1.  **Quarterly Updates:**  Updates are processed on a quarterly basis.
2.  **Direct GitHub Editing:** You can submit pull requests by editing files directly in GitHub.
3.  **Data Files:** All data is in the `csrankings-[a-z].csv` files, organized alphabetically by faculty first name initial.
4.  **Detailed Instructions:** Read the [`CONTRIBUTING.md`](CONTRIBUTING.md) for comprehensive contribution guidelines.

### Shallow Clone for Quick Contributions

To contribute without a full local clone:

1.  Fork the CSrankings repository.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes, push to your fork, and create a pull request.

## Trying it out at Home

### Prerequisites

*   Install the following:
    ```bash
    apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
    ```
*   Install [pypy](https://doc.pypy.org/en/latest/install.html)

### Building and Running

1.  Download the DBLP data: `make update-dblp` (requires ~19GiB memory)
2.  Build databases: `make`
3.  Test locally: run a web server (e.g., `python3 -m http.server`) and connect to `http://0.0.0.0:8000`

## Acknowledgements

This site was developed primarily by [Emery Berger](https://emeryberger.com). It incorporates code and data from Swarat Chaudhuri, Papoutsaki et al., and DBLP.org (under the ODC Attribution License).