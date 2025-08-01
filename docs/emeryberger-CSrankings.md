# CSrankings: The Premier Ranking of Computer Science Schools

**CSrankings provides a data-driven, metrics-based ranking of top computer science schools, offering a transparent and objective view of research productivity.** ([Original Repo](https://github.com/emeryberger/CSrankings))

## Key Features:

*   **Metrics-Based Ranking:** Ranks institutions based on the number of publications by faculty at top computer science conferences, ensuring an objective assessment.
*   **Difficult-to-Game Approach:** Employs a methodology designed to be resistant to manipulation, unlike rankings based solely on surveys or citations.
*   **Data-Driven:** Leverages publication data to provide a clear picture of research activity across different areas of computer science.
*   **Open and Transparent:** The codebase and data are publicly available, allowing for scrutiny and community contributions.
*   **Community Contributions:**  Supports contributions to refine and expand the data. (See `CONTRIBUTING.md` for more details)

## Contributing

### Adding or Modifying Affiliations

Updates are processed quarterly. You can submit pull requests anytime, but they may not be processed until the next quarter. You can edit files directly in GitHub to create pull requests. All data is in the files `csrankings-[a-z].csv`, with authors listed alphabetically by their first name, organized by the initial letter. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for full details on how to contribute.

## Running Locally

To run this site locally:

1.  Download the DBLP data by running `make update-dblp`.
2.  Rebuild the databases by running `make`.
3.  Test by running a local web server (e.g., `python3 -m http.server`) and connecting to [http://0.0.0.0:8000](http://0.0.0.0:8000).

You will also need to install the necessary dependencies, which includes: `libxml2-utils`, `npm`, `typescript`, `closure-compiler`, `python-lxml`, [pypy](https://doc.pypy.org/en/latest/install.html), and `basex`. You can install them using:

```bash
apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
```

### Quick Contribution via a Shallow Clone

To contribute without a full local clone:

1.  Fork the CSrankings repo.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes on a branch, push to your clone, and create a pull request.

## Acknowledgements

This site was developed primarily by and is maintained by [Emery Berger](https://emeryberger.com). It incorporates extensive feedback from many contributors.

This site was initially based on code and data collected by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) (UT-Austin). The original faculty affiliation dataset was constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

This site uses information from [DBLP.org](http://dblp.org) which is made available under the ODC Attribution License.

## License

CSRankings is covered by the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).