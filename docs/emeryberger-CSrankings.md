# CSrankings: Metrics-Based Computer Science School Rankings

**Discover the top computer science schools based on research publication performance, avoiding the subjectivity of surveys.**

[View the original repository on GitHub](https://github.com/emeryberger/CSrankings)

This project provides a data-driven ranking of computer science schools, focusing on research output in key areas. Unlike rankings based on surveys, CSrankings leverages publication data from the most selective conferences to provide a more objective and difficult-to-game evaluation of research excellence.

## Key Features:

*   **Metrics-Based Ranking:**  Relies on publication counts in top computer science conferences, offering a more objective assessment.
*   **Areas of Computer Science:** Tracks and ranks institutions across various computer science research areas.
*   **Data-Driven Approach:** Utilizes a comprehensive dataset from DBLP and other sources to ensure data accuracy and relevance.
*   **Community-Driven:**  Leverages community contributions for data maintenance and improvement.
*   **Transparent Methodology:**  Detailed methodology and FAQ available for complete understanding.
*   **Easy to Contribute:** Offers clear instructions on how to add or modify affiliations.

## Contributing

### Adding or Modifying Affiliations

**_NOTE: Updates are now processed on a quarterly basis._**

You can contribute to the CSrankings data by submitting pull requests. All data is stored in CSV files (`csrankings-[a-z].csv`), alphabetized by first name. Please review `CONTRIBUTING.md` for comprehensive contribution guidelines.

### Shallow Clone Contribution

A shallow clone is recommended for quick contribution:

1.  Fork the CSrankings repository on GitHub.
2.  Clone your fork using a shallow clone: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes on a branch, push them to your fork, and create a pull request.

## Running Locally

To run the CSrankings site locally, follow these steps:

1.  Download the DBLP data: `make update-dblp` (requires ~19GB of memory)
2.  Rebuild the databases: `make`
3.  Run a local web server: `python3 -m http.server`
4.  Access the site:  `http://0.0.0.0:8000`

### Dependencies:

You'll need to install: `libxml2-utils`, `npm`, `typescript`, `google-closure-compiler`, `python-lxml`, `pypy`, and `basex`. Then run the following:

`apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler`

## Acknowledgements

CSrankings was developed primarily by [Emery Berger](https://emeryberger.com) and incorporates contributions from a community of researchers and developers. The project builds upon the work of Swarat Chaudhuri (UT-Austin), Papoutsaki et al., and utilizes data from DBLP.org, licensed under the ODC Attribution License.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).