# CS Rankings: The Data-Driven Guide to Top Computer Science Schools

**Looking for the definitive ranking of top computer science programs?** This project provides a comprehensive, metrics-based ranking of computer science schools, meticulously crafted using publication data from leading conferences. Explore the live website at [https://csrankings.org](https://csrankings.org) and dive into the code that powers it on GitHub!

[Link to Original Repository: https://github.com/emeryberger/CSrankings](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Metrics-Driven Ranking:** Unlike rankings based on surveys, CS Rankings relies on the number of publications by faculty in highly selective computer science conferences.
*   **Difficult to Game:** This approach focuses on publications in top conferences, making it harder to manipulate compared to citation-based metrics.
*   **Comprehensive Data:** The project utilizes data from DBLP.org and a dataset of faculty affiliations.
*   **Community Driven:**  The project thrives on community contributions for adding and modifying affiliations.  Contributions are processed on a quarterly basis.

## Contributing

Help improve the accuracy and coverage of the rankings!  Contributions are welcome, particularly for:

*   Adding or modifying faculty affiliations.
*   Updating faculty home pages.
*   General improvements to the code or data.

Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed instructions on how to contribute.  You can edit files directly in GitHub to create pull requests.  All data is in the `csrankings-[a-z].csv` files, with authors listed alphabetically by their first name.

## Running the Site Locally

To run the site locally, you'll need to:

1.  Download the DBLP data: `make update-dblp` (requires ~19GiB memory).
2.  Rebuild the databases: `make`
3.  Run a local web server (e.g., `python3 -m http.server`) and access it at [http://0.0.0.0:8000](http://0.0.0.0:8000).

You'll also need to install the following dependencies:

`apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler`

## Quick Contribution via a Shallow Clone

For making changes without a full clone, follow these steps:

1.  Fork the CSrankings repo.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes on a branch, push to your clone, and create a pull request.

## Acknowledgements

This project was developed primarily by [Emery Berger](https://emeryberger.com) and benefits from extensive community contributions. It's based on work by Swarat Chaudhuri and Papoutsaki et al., and uses data from DBLP.org.

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).