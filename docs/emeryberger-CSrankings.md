# CSrankings: Your Go-To Ranking of Top Computer Science Schools

**CSrankings provides a data-driven ranking of computer science schools based on faculty publications in leading conferences, offering a robust and objective assessment.**  Explore the most relevant and up-to-date rankings of CS programs using this open-source project. ([See the original repository](https://github.com/emeryberger/CSrankings))

## Key Features:

*   **Metrics-Based Ranking:**  Ranks institutions based on faculty publications in the most selective computer science conferences, providing a transparent and difficult-to-game evaluation.
*   **Data-Driven Approach:**  Unlike survey-based rankings, CSrankings relies on objective, quantifiable data, ensuring a fair and accurate assessment of research productivity.
*   **Comprehensive Scope:**  Covers a wide range of computer science areas, allowing for a detailed comparison of institutions and faculty expertise.
*   **Community-Driven:**  The project is open-source, allowing contributions from the community to maintain data accuracy and relevance.
*   **Regular Updates:**  Data is updated quarterly to reflect the latest research output.

## How CSrankings Works

CSrankings utilizes a data-driven approach to rank universities by analyzing faculty publications in top-tier computer science conferences. This method focuses on research output and faculty participation, providing a reliable assessment of a school's research strengths.

## Contribute to CSrankings

Want to help improve the rankings?  You can contribute by:

*   Adding or modifying faculty affiliations
*   Updating data
*   Suggesting improvements

To contribute, submit pull requests with your updates.  Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file within the repository for detailed instructions.  Changes are processed on a quarterly basis.

## Technical Details

### Running Locally

To run the site locally, follow these steps:

1.  Download DBLP data: `make update-dblp` (requires ~19GiB memory)
2.  Rebuild databases: `make`
3.  Run a local web server: `python3 -m http.server`
4.  Access the site: [http://0.0.0.0:8000](http://0.0.0.0:8000)

### Dependencies

You will need to install the following:

*   `libxml2-utils` (or equivalent)
*   `npm`
*   `typescript`
*   `google-closure-compiler`
*   `python-lxml`
*   `pypy`
*   `basex`

Use a command like:
``apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler``

### Quick Contribution via a Shallow Clone

1.  Fork the CSrankings repo.
2.  Shallow clone your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes, commit to a branch, push, and create a pull request.

## Acknowledgements

CSrankings was developed primarily by [Emery Berger](https://emeryberger.com) and incorporates contributions from a wide community of researchers and contributors.  It draws upon data from DBLP.org, licensed under the ODC Attribution License. The original faculty affiliation dataset was constructed by Papoutsaki et al.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).