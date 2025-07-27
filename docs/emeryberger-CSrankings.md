# CS Rankings: A Metrics-Driven Ranking of Top Computer Science Schools

This project provides a comprehensive, metrics-based ranking of computer science schools, offering a data-driven approach to evaluating research excellence. **Explore the cutting edge of computer science research with CS Rankings!**

**[View the original repository on GitHub](https://github.com/emeryberger/CSrankings)**

## Key Features:

*   **Metrics-Based Approach:** Unlike rankings based on surveys, this ranking uses a data-driven, metrics-based approach, measuring the number of publications by faculty at top computer science conferences.
*   **Focus on Research Excellence:** The ranking highlights institutions and faculty actively engaged in research across various areas of computer science.
*   **Difficult to Game:** The methodology focuses on publications in highly selective conferences, making it more difficult to manipulate compared to citation-based metrics.
*   **Open Data and Contribution:** The project utilizes open data and welcomes community contributions to improve and maintain the ranking.
*   **Quarterly Updates:** Data is updated quarterly, ensuring the ranking reflects the latest research activities.

## Contributing:

You can contribute to the CS Rankings project by submitting pull requests to update faculty affiliations and other data. Please read the `CONTRIBUTING.md` file for detailed instructions on how to contribute. **Note that updates are processed on a quarterly basis.**

### Getting Started:

1.  **Data Files:** The core data is stored in `csrankings-[a-z].csv` files.
2.  **Shallow Clone for Contributions:** To contribute without a full clone, use a shallow clone of your fork.

### Setting up the Environment

To run the site locally, you will need to download the DBLP data, rebuild the databases, and run a local web server. You will also need to install the following dependencies:

*   `libxml2-utils` (or the package containing `xmllint`)
*   `npm`
*   `typescript`
*   `google-closure-compiler`
*   `python-lxml`
*   `pypy`
*   `basex`

Install dependencies using this command:

```bash
apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
```

## Acknowledgements

This project was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). It builds upon the work of Swarat Chaudhuri, Papoutsaki et al., and utilizes data from DBLP.org, which is made available under the ODC Attribution License.

## License

CS Rankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).