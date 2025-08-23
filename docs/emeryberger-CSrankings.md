# CSrankings: The Premier Computer Science School Ranking (and Repository)

**CSrankings provides a metrics-based ranking of top computer science schools, built on faculty publications at leading conferences.** Unlike rankings based on surveys, CSrankings uses a rigorous, objective methodology to evaluate institutions and faculty research contributions. This repository houses the code and data behind the CSrankings website, allowing you to contribute and explore the data. (See the live rankings at: [https://csrankings.org](https://csrankings.org))  

[View the original repository on GitHub](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Metrics-Driven Ranking:**  CSrankings uses a data-driven approach based on faculty publications in top-tier computer science conferences.
*   **Objective Evaluation:**  This ranking avoids subjective surveys, focusing on measurable research output.
*   **Community Driven:**  Contribute to the ranking by adding or modifying affiliations and helping to maintain the data.
*   **Open Source:**  The code and data are available for anyone to explore, modify, and contribute to.
*   **Up-to-date:** The database is updated quarterly, keeping pace with the ever-changing world of research.

## Contributing

Want to contribute to CSrankings?  You can contribute by:

*   **Adding or Modifying Affiliations:**  All data is in the `csrankings-[a-z].csv` files. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
*   **Submitting Pull Requests:** Your pull requests can be submitted at any time.  However, updates are processed quarterly.

### Shallow Clone for Easy Contribution

1.  Fork the CSrankings repo.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes on a branch, push them to your clone, and create a pull request on GitHub.

## Running Locally

To run CSrankings locally, follow these steps:

1.  Download DBLP data: `make update-dblp` (requires ~19GB memory).
2.  Rebuild databases: `make`.
3.  Test locally: Run a web server (e.g., `python3 -m http.server`) and connect to [http://0.0.0.0:8000](http://0.0.0.0:8000).
4.  Install Dependencies:

    ```bash
    apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
    ```

## Acknowledgements

CSrankings was developed and is primarily maintained by [Emery Berger](https://emeryberger.com). It builds on the work of Swarat Chaudhuri and the faculty dataset constructed by Papoutsaki et al., along with the use of DBLP.org data, which is made available under the ODC Attribution License.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).