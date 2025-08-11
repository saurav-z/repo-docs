# CSrankings: Your Definitive Guide to Top Computer Science Schools

**CSrankings is a metrics-driven ranking of computer science schools, providing a data-focused and objective assessment of research performance.** Unlike rankings based on surveys, CSrankings relies on the number of publications by faculty at the most selective conferences in each area of computer science, offering a reliable and current view of top institutions.  Explore the repository on [GitHub](https://github.com/emeryberger/CSrankings) for code and data.

## Key Features

*   **Metrics-Based:** Rankings are determined by faculty publications in top-tier computer science conferences.
*   **Objective:** The methodology avoids the subjectivity of survey-based rankings.
*   **Areas of Computer Science:** Covers a broad range of CS research areas.
*   **Regular Updates:** Data is updated quarterly to reflect the latest research outputs.
*   **Community Driven:**  Contributions are welcome; a process is defined to add or modify school affiliations.

## How CSrankings Works

The ranking system uses publication data from the DBLP database to evaluate faculty research output. This approach is designed to be resistant to manipulation, focusing on publications in highly selective conferences. See the [FAQ](https://csrankings.org/faq.html) for details.

## Contributing

You can contribute to CSrankings by adding or modifying affiliations.

*   **Data Files:**  Affiliations are stored in `csrankings-[a-z].csv` files.
*   **Contribution Guidelines:**  Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed instructions.
*   **Pull Requests:** Submit your changes as pull requests on GitHub.

## Setting Up Locally

To run the site locally, follow these steps:

1.  Download the DBLP data using `make update-dblp` (requires ~19GB memory).
2.  Rebuild the databases with `make`.
3.  Run a local web server (e.g., `python3 -m http.server`).
4.  Access the site at [http://0.0.0.0:8000](http://0.0.0.0:8000).

**Dependencies:**  You'll need `libxml2-utils`, `npm`, `typescript`, `closure-compiler`, `python-lxml`, [pypy](https://doc.pypy.org/en/latest/install.html), and `basex`.  Install them using the command provided in the original README:

```bash
apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
```

## Shallow Clone for Contribution

To contribute without a full clone:

1.  Fork the CSrankings repository.
2.  Create a shallow clone: `git clone --depth 1 https://github.com/yourusername/CSrankings`.
3.  Make your changes on a branch, push, and create a pull request.

## Acknowledgements

CSrankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). This work has been made possible with extensive feedback from numerous contributors, the original code and data from [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/), and the faculty affiliation dataset from [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html). This project uses information from [DBLP.org](http://dblp.org).

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).