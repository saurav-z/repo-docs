# CSrankings: Explore and Compare Top Computer Science Programs

**CSrankings.org** provides a data-driven ranking of computer science departments worldwide, empowering students, researchers, and industry professionals to make informed decisions.  Explore the source code and contribute to the project on [GitHub](https://github.com/emeryberger/CSrankings).

## Key Features:

*   **Metrics-Based Ranking:** Uses publication counts in highly selective conferences to rank CS departments, avoiding survey-based methodologies.
*   **Transparent and Data-Driven:**  Provides a transparent and objective ranking based on research output, avoiding manipulation.
*   **Comprehensive Coverage:**  Tracks faculty research activity across various computer science areas.
*   **Open Source and Collaborative:**  The codebase is open-source, allowing community contributions to improve data accuracy and expand functionality.
*   **Easy-to-Understand Data:** Provides access to the codebase and data used for building the website.

## How CSrankings Works

Unlike rankings based on subjective surveys, CSrankings relies on objective metrics. The primary method is to measure the number of publications by faculty in top computer science conferences.

## Contribute to CSrankings

The project welcomes contributions!

### Adding or Modifying Affiliations

*   Updates are processed quarterly.
*   Data is stored in `csrankings-[a-z].csv` files, organized by first name initial.
*   Follow the guidelines in `CONTRIBUTING.md` to submit pull requests.

### Setting up Locally

To run CSrankings locally, you'll need:

*   DBLP data (download using `make update-dblp` - requires ~19GB of memory).
*   Dependencies: `libxml2-utils`, `npm`, `typescript`, `closure-compiler`, `python-lxml`, `pypy`, and `basex`.

Run `make` to rebuild the databases. Then, test the website with a local server (e.g., `python3 -m http.server`).

### Shallow Clone Contribution

For quick contributions without a full clone:

1.  Fork the CSrankings repository.
2.  Perform a shallow clone: `git clone --depth 1 https://github.com/yourusername/CSrankings`.
3.  Make your changes, push to your clone, and submit a pull request.

## Acknowledgements

CSrankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). It's based on data and code initially collected by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) (UT-Austin), and the original faculty affiliation dataset was constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).