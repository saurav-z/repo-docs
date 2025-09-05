# CSrankings: The Premier Computer Science School Ranking Website

**CSrankings provides a data-driven, metric-based ranking of top computer science schools, offering a transparent and objective assessment of research excellence.** ([Original Repository](https://github.com/emeryberger/CSrankings))

## Key Features

*   **Metrics-Based Approach:** Unlike rankings based on surveys, CSrankings relies on objective metrics, specifically, the number of publications by faculty at the most selective computer science conferences.
*   **Difficult-to-Game Methodology:** The focus on publication counts in highly selective conferences makes the ranking resistant to manipulation and gaming.
*   **Comprehensive Data:** CSrankings incorporates data from DBLP.org and a dataset developed by Papoutsaki et al., providing a rich and extensive dataset for analysis.
*   **Open Source:** Contribute to and improve the ranking by contributing to the data and code.
*   **Community Driven:** Maintained and improved with help from many contributors who help add and maintain faculty affiliations and more.

## Contributing

### How to Contribute
*   **Quarterly Updates:** Updates are processed quarterly.
*   **File-Based Contributions:** Data is stored in `csrankings-[a-z].csv` files, organized alphabetically by first name.
*   **Shallow Cloning:** You can use shallow clones for easier contribution.
*   Read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

### Running Locally

*   Install Dependencies: libxml2-utils, npm, typescript, closure-compiler, python-lxml, pypy, and basex.
*   Clone the repository.
*   Run `make update-dblp` to download DBLP data (requires ~19GB memory).
*   Run `make` to rebuild databases.
*   Test with a local web server (e.g., `python3 -m http.server`) and access at `http://0.0.0.0:8000`.

## Acknowledgements

CSrankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). It is based on code and data initially collected by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/), and the original faculty affiliation dataset was constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).