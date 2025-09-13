# CS Rankings: Your Go-To Resource for Data-Driven Computer Science School Rankings

Tired of rankings based on surveys? **CS Rankings offers a transparent, metrics-driven ranking of top computer science schools worldwide, based on faculty research output at premier conferences.**  This repository houses the code and data behind the popular [CS Rankings website](https://csrankings.org), providing a valuable resource for prospective students, researchers, and anyone interested in the landscape of computer science.

[Link to Original Repository:  https://github.com/emeryberger/CSrankings](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Metrics-Based Rankings:**  Ranks institutions based on the number of publications by faculty at the most selective computer science conferences, offering a more objective assessment than survey-based methods.
*   **Transparent Data:**  All data and code used to generate the rankings are available in this repository, allowing for complete transparency and reproducibility.
*   **Community-Driven:**  The project welcomes contributions to add or modify faculty affiliations and other data, ensuring the rankings remain up-to-date.
*   **Easy to Contribute:**  The repository provides clear instructions on how to contribute, including options for shallow clones for faster contribution.
*   **Regular Updates:** Rankings are updated quarterly.

## Contributing

The data is in `csrankings-[a-z].csv`, with authors listed in alphabetical order by their first name, organized by the initial letter.  Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for full details on how to contribute.

## Running the Site Locally

To run the site locally, you'll need to download the DBLP data and build the databases:

1.  Run `make update-dblp` (requires ~19GB of memory).
2.  Run `make` to rebuild databases.
3.  Start a local web server (e.g., `python3 -m http.server`) and connect to [http://0.0.0.0:8000](http://0.0.0.0:8000).

**Required Dependencies:** libxml2-utils (or equivalent), npm, typescript, closure-compiler, python-lxml, [pypy](https://doc.pypy.org/en/latest/install.html), and basex. Install via:
`apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler`

## Contributing via Shallow Clone (Quick Method)

1.  Fork the CSrankings repo.
2.  Shallow clone your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`.
3.  Make changes on a branch, push them, and create a pull request. Repeat steps 1-3 when making another contribution later.

## Acknowledgements

Developed and maintained by [Emery Berger](https://emeryberger.com).  This project builds upon the work of [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) and the faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html), as well as extensive community feedback.

This site uses information from [DBLP.org](http://dblp.org) which is made available under the ODC Attribution License.

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).