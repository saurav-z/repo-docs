# CSrankings: The Premier Ranking of Computer Science Schools

**CSrankings provides a data-driven, objective ranking of computer science departments worldwide, based on faculty publication records at top conferences.** Learn more and contribute to the project at the [original repository](https://github.com/emeryberger/CSrankings).

## Key Features

*   **Metrics-Based Ranking:** Unlike rankings based on surveys, CSrankings uses a purely metrics-driven approach, focusing on the number of publications by faculty in highly selective computer science conferences.
*   **Difficult-to-Game Methodology:** The ranking system is designed to be robust against manipulation, prioritizing publications in top-tier conferences rather than citation-based metrics that are easier to exploit.
*   **Regular Updates:** The ranking and associated data are updated quarterly.
*   **Open Source and Community-Driven:** Contribute to the project by adding or modifying affiliations through pull requests. See `CONTRIBUTING.md` for details.
*   **Data Source:** Leverages data from DBLP.org, ensuring a comprehensive and up-to-date dataset.

## How to Contribute

### Adding or Modifying Affiliations

Updates are processed quarterly. You can submit pull requests at any time, but they may not be processed until the next quarter.

1.  **Edit Directly on GitHub:** You can now edit the `csrankings-[a-z].csv` files directly within GitHub.
2.  **File Structure:** Faculty are listed alphabetically by their first name, organized by the initial letter in `csrankings-[a-z].csv` files.
3.  **Contribution Guidelines:** Please review the `CONTRIBUTING.md` file for detailed instructions on how to contribute.

### Trying it out at Home

1.  **DBLP Data:** Download the DBLP data by running `make update-dblp` (requires significant memory).
2.  **Build Databases:** Run `make`.
3.  **Local Server:** Test locally using a web server (e.g., `python3 -m http.server`) and connecting to `http://0.0.0.0:8000`.
4.  **Dependencies:** Install necessary dependencies, including `libxml2-utils`, `npm`, `typescript`, `closure-compiler`, `python-lxml`, `pypy`, and `basex`:
    ```bash
    apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
    ```

### Quick Contribution via a Shallow Clone

1.  **Fork:** Fork the CSrankings repository.
2.  **Shallow Clone:** Perform a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`.
3.  **Create Branch:** Make your changes on a new branch.
4.  **Push and Pull Request:** Push your changes and create a pull request on GitHub.

## Acknowledgements

This project was developed primarily by [Emery Berger](https://emeryberger.com). It builds upon work from [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/), and incorporates the faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).