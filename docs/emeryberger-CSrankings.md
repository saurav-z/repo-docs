# CSrankings: The Premier Metric-Based Computer Science Rankings

**CSrankings** provides a data-driven, objective ranking of computer science departments and faculty, focusing on research publication performance in top-tier conferences. Find the original repository [here](https://github.com/emeryberger/CSrankings).

## Key Features:

*   **Metrics-Based Approach:** Ranks institutions and faculty based on publications at highly selective computer science conferences.
*   **Objective and Data-Driven:** Uses a purely metrics-based approach, avoiding subjective surveys and focusing on measurable research output.
*   **Difficult to Game:**  Employs publication data from top conferences, making manipulation more challenging than citation-based metrics.
*   **Open Source:** All code and data are available for review and contribution.
*   **Regular Updates:** Data is updated quarterly to reflect the latest research publications.

## Contributing

### Adding or Modifying Affiliations

Updates are processed quarterly.  Contribute by:

*   Editing `csrankings-[a-z].csv` files, with authors listed alphabetically by their first name.
*   Reviewing the [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.
*   Submitting pull requests with your changes.

### Quick Contribution (Shallow Clone)

To contribute without a full clone:

1.  Fork the CSrankings repository.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`.
3.  Make changes on a branch, push to your clone, and create a pull request.

## Setting Up Locally

To run the site locally:

1.  Run `make update-dblp` to download DBLP data (requires ~19GB memory).
2.  Run `make` to rebuild the databases.
3.  Run a local web server (e.g., `python3 -m http.server`) and access at `http://0.0.0.0:8000`.

**Required Dependencies:** libxml2-utils, npm, typescript, closure-compiler, python-lxml, pypy, basex.  Install with: `apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler`.

## Acknowledgements

Developed primarily by [Emery Berger](https://emeryberger.com).  This project is based on the work of [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) and the dataset by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).  Uses information from [DBLP.org](http://dblp.org).

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).