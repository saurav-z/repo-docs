# CS Rankings: Metrics-Based Ranking of Computer Science Schools

Tired of rankings based on surveys? **CS Rankings offers a data-driven approach to assess the research performance of computer science institutions and faculty, based on publications at top conferences.** Find the original repository [here](https://github.com/emeryberger/CSrankings).

## Key Features

*   **Metrics-Driven:** Based on the number of publications at the most selective computer science conferences, providing an objective assessment.
*   **Difficult to Game:** Employs a methodology designed to be resistant to manipulation, unlike citation-based metrics.
*   **Comprehensive Data:** Leverages data from DBLP.org to provide a broad overview of research activity.
*   **Community Driven:**  Contributions are welcome. Instructions for adding or modifying affiliations are available.
*   **Open Source:**  The code and data are publicly available for review and contribution.

## Contributing to CS Rankings

### Adding or Modifying Affiliations

**Note:** Updates are processed quarterly. Submit pull requests at any time, but they may not be processed until the next quarter.

*   Edit files directly in GitHub to create pull requests.
*   Data is in `csrankings-[a-z].csv` files, with authors alphabetized by first name.
*   Refer to `CONTRIBUTING.md` for detailed contribution instructions.

### Setting Up Locally

1.  **Download DBLP data:**  Run `make update-dblp`. (Requires ~19GiB memory).
2.  **Build the databases:**  Run `make`.
3.  **Test Locally:** Run a local web server (e.g., `python3 -m http.server`) and access it via `http://0.0.0.0:8000`.
4.  **Install Dependencies:**

    ```bash
    apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
    ```

    Also, install [pypy](https://doc.pypy.org/en/latest/install.html).

### Quick Contribution via Shallow Clone

To contribute without a full clone:

1.  Fork the CSrankings repository.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes on a branch, push, and create a pull request.

## Acknowledgements

Developed primarily by [Emery Berger](https://emeryberger.com), incorporating feedback from numerous contributors.  Based on initial code and data collected by Swarat Chaudhuri and the faculty affiliation dataset constructed by Papoutsaki et al.

Data from [DBLP.org](http://dblp.org) is used, licensed under the ODC Attribution License.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).