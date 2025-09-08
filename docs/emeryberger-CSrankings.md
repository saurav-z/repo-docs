# CSrankings: A Data-Driven Ranking of Top Computer Science Schools

**CSrankings provides a unique, metrics-based ranking of computer science departments, focusing on research output at top conferences.**

[View the original repository on GitHub](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Metrics-Based Ranking:**  Unlike rankings based on surveys, CSrankings uses a data-driven approach, measuring faculty publications in highly selective computer science conferences.
*   **Focus on Research Excellence:**  The ranking emphasizes research activity and impact within specific computer science areas.
*   **Difficult to Game:** The methodology is designed to be resistant to manipulation, focusing on publications in prestigious venues.
*   **Comprehensive Data Source:** Leverages data from DBLP.org to build a comprehensive ranking.
*   **Community-Driven:** The project welcomes contributions to update and improve the data.

## Contributing

You can contribute by:

*   **Adding or modifying affiliations:** Updates are processed quarterly.  See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.
*   **Quick contribution via a shallow clone:** Use a shallow clone for faster contributions. Instructions are provided in the README.

## Technical Details

### Running the Site Locally

To run the site locally, you'll need:

*   DBLP data (download using `make update-dblp`, requires ~19GB of memory)
*   Build the databases (`make`)
*   A local web server (e.g., `python3 -m http.server`)

### Dependencies

You will need to install the following dependencies:

*   libxml2-utils (or equivalent)
*   npm
*   typescript
*   google-closure-compiler
*   python-lxml
*   pypy
*   basex

Install them via a command line like:
``apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler``

## Acknowledgements

Developed primarily by [Emery Berger](https://emeryberger.com).

Based on code and data collected by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) and the faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

This site uses information from [DBLP.org](http://dblp.org) which is made available under the ODC Attribution License.

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).