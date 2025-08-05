# CSrankings: A Data-Driven Ranking of Computer Science Schools

CSrankings provides a meticulously crafted, metric-based ranking of top computer science schools, offering a data-driven perspective on research excellence.  Explore the code and data behind this innovative ranking system and contribute to its continued development at the original repository: [https://github.com/emeryberger/CSrankings](https://github.com/emeryberger/CSrankings).

## Key Features

*   **Metric-Based Approach:** Unlike rankings based on surveys, CSrankings relies entirely on the number of publications by faculty in top-tier computer science conferences, offering a robust and objective evaluation.
*   **Difficult-to-Game Methodology:**  This ranking system prioritizes publications in highly selective conferences, making it challenging to manipulate the results compared to citation-based metrics.
*   **Open-Source Data and Code:** The complete codebase and data are available for public access, enabling users to understand the ranking methodology, and even contribute to its evolution.
*   **Regular Updates:** The ranking and underlying data are regularly updated to reflect the latest research and advancements in the field.

## Contributing

Contributions to CSrankings are welcome! You can contribute by:

*   **Adding or Modifying Affiliations:**  Update faculty affiliations and homepages.  Changes are processed quarterly.
*   **Submitting Pull Requests:**  You can directly edit files in GitHub to submit changes. Please refer to `CONTRIBUTING.md` for detailed instructions.
*   **Data Files:**  Data is stored in `csrankings-[a-z].csv` files, organized alphabetically by first name.

## Running the Project Locally

To run CSrankings locally, follow these steps:

1.  Download the DBLP data: `make update-dblp` (requires ~19GiB memory).
2.  Rebuild the databases: `make`.
3.  Test the site: Run a local web server (e.g., `python3 -m http.server`) and access it at [http://0.0.0.0:8000](http://0.0.0.0:8000).
4.  Install dependencies (libxml2-utils, npm, typescript, etc.).

## Quick Contribution via Shallow Clone

For those who wish to contribute without a full clone, use a shallow clone:

1.  Fork the CSrankings repository.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`.
3.  Make changes, push to your fork, and create a pull request.

## Acknowledgements

CSrankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com).  It builds upon the work of [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) (UT-Austin) and the faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).  The project incorporates data from [DBLP.org](http://dblp.org) which is available under the ODC Attribution License.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).