# CSrankings: Ranking Top Computer Science Schools

**CSrankings.org provides a data-driven ranking of computer science institutions and faculty based on publication records.** This approach offers a more objective view than traditional survey-based methods.  You can explore the project further on [GitHub](https://github.com/emeryberger/CSrankings).

## Key Features:

*   **Metrics-Based Ranking:**  Ranks institutions and faculty based on publications in top-tier computer science conferences, avoiding subjective survey-based approaches.
*   **Difficult to Game:** Relies on publications in highly selective conferences, making the ranking more resistant to manipulation.
*   **Data-Driven:** Leverages publicly available data to provide a transparent and objective assessment.
*   **Community-Driven:**  The project welcomes contributions to maintain and update the rankings.

## Contributing

### Adding or modifying affiliations

**_NOTE: Updates are now processed on a quarterly basis. You may submit pull requests at any time, but they may not be processed until the next quarter (after three months have elapsed)._**

You can now edit files directly in GitHub to create pull requests. All data is
in the files `csrankings-[a-z].csv`, with authors listed in
alphabetical order by their first name, organized by the initial letter. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for full details on how to contribute.

### Quick contribution via a shallow clone

To contribute a change without creating a full local clone of the
CSrankings repo, you can do a shallow clone. To do so, follow these
steps:

1. Fork the CSrankings repo. If you have an existing fork, but it is
not up to date with the main repository, this technique may not
work. If necessary, delete and re-create your fork to get it up to
date. (Do not delete your existing fork if it has unmerged changes you
want to preserve!)
2. Do a shallow clone of your fork: `git clone --depth 1
https://github.com/yourusername/CSrankings`. This will only download
the most recent commit, not the full git history.
3. Make your changes on a branch, push them to your clone, and create
a pull request on GitHub as usual.

If you want to make another contribution and some time has passed,
perform steps 1-3 again, creating a fresh fork and shallow clone.

## Running Locally

To run the site locally, follow these steps:

1.  Download the DBLP data: `make update-dblp` (requires ~19GB memory).
2.  Rebuild databases: `make`.
3.  Run a local web server: `python3 -m http.server`.
4.  Access the site in your browser: [http://0.0.0.0:8000](http://0.0.0.0:8000).

### Dependencies

You will need to install the following dependencies:

```bash
apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
```
You will also need: [pypy](https://doc.pypy.org/en/latest/install.html)

## Acknowledgements

This site was developed by and is maintained by [Emery Berger](https://emeryberger.com). It builds upon code and data collected by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) and the original faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html). It also uses information from [DBLP.org](http://dblp.org).

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).