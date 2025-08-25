# CSrankings: The Premier Computer Science School Ranking

**CSrankings provides a data-driven, objective ranking of top computer science schools based on faculty publications in highly selective conferences.**  This repository contains the code and data that power the website, offering a transparent and metric-based approach to evaluate CS programs.  Access the live rankings at [https://csrankings.org](https://csrankings.org) or explore the source code at [https://github.com/emeryberger/CSrankings](https://github.com/emeryberger/CSrankings).

## Key Features:

*   **Metric-Based Ranking:** Unlike rankings based on surveys, CSrankings uses a data-driven approach, evaluating schools based on faculty publications in top-tier computer science conferences.
*   **Difficult-to-Game Methodology:**  This approach is designed to be difficult to manipulate, providing a more reliable assessment of research productivity.
*   **Transparent Data Source:** Based on information from DBLP.org, made available under the ODC Attribution License.
*   **Community-Driven:** Contributions are welcomed to help maintain and update faculty affiliations.
*   **Open Source:**  The code is available on GitHub, encouraging transparency and collaboration.

## Contributing

*   **Quarterly Updates:**  Updates to the rankings are processed quarterly.
*   **How to Contribute:**  Submit pull requests directly in GitHub by editing the `csrankings-[a-z].csv` files.  See `CONTRIBUTING.md` for detailed instructions.
*   **Shallow Clone Option:**  For quicker contributions without a full repository clone, use a shallow clone to make and submit changes.

##  Setting Up Locally

To run CSrankings locally, follow these steps:

1.  Download DBLP data: `make update-dblp`
2.  Build the databases: `make`
3.  Run a local web server (e.g., `python3 -m http.server`)
4.  Access the site at:  `http://0.0.0.0:8000`

**Required Dependencies:** `libxml2-utils`, `npm`, `typescript`, `closure-compiler`, `python-lxml`, `pypy`, `basex`

**Install Dependencies:**  `apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler`

## Acknowledgements

CSrankings was primarily developed by [Emery Berger](https://emeryberger.com). It builds upon the work of [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) and the faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).