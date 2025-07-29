# CS Rankings: Your Definitive Guide to Top Computer Science Schools

**Looking to identify the leading computer science institutions and faculty?** CS Rankings provides a data-driven, metric-based ranking system focused on publications at top-tier computer science conferences.  This approach offers a transparent, reliable, and difficult-to-manipulate evaluation of research excellence.

[Visit the original repository on GitHub](https://github.com/emeryberger/CSrankings)

## Key Features

*   **Metric-Based Ranking:** Rankings are based on the number of publications by faculty at the most selective computer science conferences.
*   **Transparent Methodology:**  Uses a clear and transparent methodology, unlike rankings based on surveys.
*   **Difficult to Game:**  Focuses on publications in top conferences, making it difficult to manipulate the rankings.
*   **Regular Updates:** The dataset is updated quarterly.
*   **Community-Driven:** Contributions are welcome for adding and modifying affiliations (see the CONTRIBUTING.md file).

## Contributing

### Adding or Modifying Affiliations
Contributions are processed quarterly. You can submit pull requests at any time.
Data is found in the `csrankings-[a-z].csv` files, with authors alphabetized by first name initial.  See the `CONTRIBUTING.md` file for more details.

### Using the Site Locally

1.  Download DBLP data: `make update-dblp` (requires ~19GB of memory).
2.  Rebuild databases: `make`
3.  Run a local web server (e.g., `python3 -m http.server`) and access it at `http://0.0.0.0:8000`.

**Required dependencies:**  `libxml2-utils`, `npm`, `typescript`, `closure-compiler`, `python-lxml`, [pypy](https://doc.pypy.org/en/latest/install.html), and `basex`.

## Quick Contribution via Shallow Clone

For quick contributions, use a shallow clone to avoid downloading the full repository history:

1.  Fork the CSrankings repo.
2.  Shallow clone your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes on a branch, push them, and create a pull request.

## Acknowledgements

*   **Developed by:** [Emery Berger](https://emeryberger.com).
*   **Based on original work by:** [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) and the faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).
*   **Data Source:** Uses information from [DBLP.org](http://dblp.org) under the ODC Attribution License.

## License

CS Rankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).