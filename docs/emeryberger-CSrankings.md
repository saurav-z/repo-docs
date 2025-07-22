# CS Rankings: The Definitive Ranking of Computer Science Schools

**CS Rankings** is a data-driven ranking of top computer science schools, based entirely on faculty publications at top conferences. Explore the source code and contribute to this valuable resource on [GitHub](https://github.com/emeryberger/CSrankings).

## Key Features:

*   **Metrics-Based Ranking:** Unlike rankings based on surveys, CS Rankings relies on the number of publications by faculty at highly selective computer science conferences.
*   **Difficult-to-Game Approach:** This methodology focuses on publication records, making it challenging to manipulate the rankings.
*   **Comprehensive Data:** Utilizes extensive data from DBLP and contributions from the community.
*   **Open Source:** The code and data are available for review, contribution, and enhancement.
*   **Regular Updates:** Data is updated quarterly to reflect the latest research and faculty affiliations.

## Contributing

CS Rankings welcomes contributions! You can add or modify affiliations by:

*   **Directly editing CSV files:**  Files are named `csrankings-[a-z].csv` and organized alphabetically by author's first name. Please read `CONTRIBUTING.md` for detailed guidelines.
*   **Submitting Pull Requests:** Contribute changes through pull requests, which are processed on a quarterly basis.
*   **Using Shallow Clone:** To contribute without a full local clone, use a shallow clone as described in the README.

## Running CS Rankings Locally

To run the site locally, you'll need to download the DBLP data (requires approximately 19GB of memory) by running ``make update-dblp`` and then build the databases by running ``make``.  You can then run a local web server (e.g., ``python3 -m http.server``) and access it via [http://0.0.0.0:8000](http://0.0.0.0:8000).

**Dependencies:** You will also need to install the following:
`libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler`
## Acknowledgements

CS Rankings was developed by Emery Berger, and incorporates feedback from numerous contributors. It builds upon the work of Swarat Chaudhuri (UT-Austin) and the faculty affiliation dataset constructed by Papoutsaki et al. It leverages data from DBLP.org, available under the ODC Attribution License.

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).