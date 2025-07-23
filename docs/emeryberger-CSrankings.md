# CSrankings: The Data-Driven Ranking of Top Computer Science Schools

**CSrankings provides a unique, metrics-based ranking of top computer science schools, focusing on research output.** Unlike rankings based on surveys, CSrankings uses a data-driven approach, measuring faculty publications in top computer science conferences.  You can find the original repository on GitHub: [https://github.com/emeryberger/CSrankings](https://github.com/emeryberger/CSrankings).

## Key Features

*   **Metrics-Based Ranking:** CSrankings focuses on the number of publications by faculty at leading computer science conferences.
*   **Difficult to Game:**  The ranking methodology aims to be resistant to manipulation compared to citation-based metrics.
*   **Open Source & Community Driven:**  Contribute to the ranking by adding or modifying affiliations.  See the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed instructions.
*   **Regular Updates:**  The rankings are updated quarterly to reflect the latest research publications.
*   **Data Source:** Utilizes data from DBLP.org, available under the ODC Attribution License.

## Contributing

### Adding or Modifying Affiliations

1.  **Quarterly Updates:** Changes are processed quarterly.
2.  **Direct Editing:** Edit data files (`csrankings-[a-z].csv`) directly in GitHub to create pull requests.
3.  **Contribution Guide:** Follow the instructions in [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

### Shallow Clone for Quick Contributions

1.  **Fork:** Fork the CSrankings repository on GitHub.
2.  **Shallow Clone:** Use `git clone --depth 1 https://github.com/yourusername/CSrankings` to clone your fork.
3.  **Create a Branch:** Make your changes on a new branch and push to your fork.
4.  **Pull Request:** Create a pull request on GitHub.

## Running Locally

To run the site locally, you'll need to:

1.  Download the DBLP data: `make update-dblp` (requires ~19GiB memory).
2.  Rebuild databases: `make`.
3.  Run a local web server (e.g., `python3 -m http.server`) and access it in your browser.

**Required Dependencies:** libxml2-utils, npm, typescript, closure-compiler, python-lxml, [pypy](https://doc.pypy.org/en/latest/install.html), and basex.

## Acknowledgements

This project was developed and is maintained by [Emery Berger](https://emeryberger.com). It builds upon the work of [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/), the faculty affiliation dataset by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html), and incorporates extensive feedback from a community of contributors.

## License

This project is covered by the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).