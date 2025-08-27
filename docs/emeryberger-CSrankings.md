# CSrankings: The Data-Driven Computer Science School Ranking

**CSrankings provides a comprehensive, data-driven ranking of computer science schools, focusing on faculty research output.** Unlike rankings based on surveys, CSrankings leverages publication data from top computer science conferences to offer a transparent and objective assessment. Explore the code and data behind this powerful ranking system on [GitHub](https://github.com/emeryberger/CSrankings).

## Key Features:

*   **Metrics-Based Ranking:** Ranks schools based on the number of publications by faculty in highly selective computer science conferences.
*   **Objective Assessment:** Avoids survey-based methods, relying on quantifiable research output.
*   **Transparency:** Code and data publicly available on GitHub for scrutiny and contribution.
*   **Regular Updates:** Data is updated regularly to reflect the latest research publications.
*   **Community Driven:** Contributions are welcome to maintain and improve the ranking.

## Contributing to CSrankings

Want to help improve the data? Here's how you can contribute:

*   **Add or Modify Affiliations:** Data is in `csrankings-[a-z].csv` files.  See `CONTRIBUTING.md` for details (updates processed quarterly).
*   **Shallow Clone for Quick Contributions:** Use a shallow clone for efficient editing (see instructions below).

### Shallow Clone Instructions

1.  Fork the CSrankings repo.
2.  Shallow clone your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make your changes on a branch, push them to your clone, and create a pull request on GitHub.

## Getting Started Locally

To run the website locally:

1.  Download DBLP data: `make update-dblp` (requires ~19GiB memory).
2.  Rebuild databases: `make`
3.  Run a local web server (e.g., `python3 -m http.server`)
4.  View the website at: `http://0.0.0.0:8000`

### Required Dependencies

You will need the following dependencies installed:

```bash
apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
```
also install [pypy](https://doc.pypy.org/en/latest/install.html)

## Acknowledgements

CSrankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com), with significant contributions from the community. The project builds on the work of [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) and the faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

This site uses data from [DBLP.org](http://dblp.org) under the ODC Attribution License.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).