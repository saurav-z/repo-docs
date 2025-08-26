# CS Rankings: The Premier Metric-Based Ranking of Computer Science Schools

**Looking for a data-driven approach to evaluate computer science programs?** CS Rankings provides a comprehensive, metrics-based ranking of top computer science schools based on faculty publications at premier conferences.  Find the original repository [here](https://github.com/emeryberger/CSrankings).

## Key Features:

*   **Metrics-Driven Ranking:**  Unlike rankings based on surveys, CS Rankings utilizes a data-driven approach, measuring the number of publications by faculty at the most selective computer science conferences.
*   **Difficult-to-Game Methodology:**  The ranking methodology focuses on publications in highly selective conferences, which is designed to be more resistant to manipulation compared to citation-based metrics.
*   **Focus on Research Excellence:**  CS Rankings highlights institutions and faculty actively engaged in cutting-edge computer science research.
*   **Open Source and Community-Driven:** The project is open-source and welcomes contributions to maintain the accuracy and comprehensiveness of the rankings.
*   **Quarterly Updates:** The data is updated quarterly to ensure the rankings reflect the latest research output.

## Contributing to CS Rankings

### Adding or Modifying Affiliations

To contribute, you can submit pull requests with changes.

**_NOTE: Updates are now processed on a quarterly basis. You may submit pull requests at any time, but they may not be processed until the next quarter (after three months have elapsed)._**

You can directly edit the data files (`csrankings-[a-z].csv`), with authors listed in alphabetical order by their first name, directly on GitHub to create pull requests. Detailed contribution guidelines are available in the [`CONTRIBUTING.md`](CONTRIBUTING.md) file.

## Running CS Rankings Locally

To run the website locally, you will need to download the DBLP data: `make update-dblp` (requires ~19GB of memory). Then rebuild databases with `make`. Access it via a local web server (e.g., `python3 -m http.server`) at [http://0.0.0.0:8000](http://0.0.0.0:8000).

### Dependencies

You will need to install the following dependencies:

```bash
apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
```

You may also need pypy.

## Quick Contribution via a Shallow Clone

For smaller contributions, you can use a shallow clone to avoid downloading the full repository history:

1.  Fork the CSrankings repo.
2.  Clone your fork with: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make your changes on a branch, push, and create a pull request.
    *   If you want to make another contribution and some time has passed,
    perform steps 1-3 again, creating a fresh fork and shallow clone.

## Acknowledgements

This site was developed primarily by and is maintained by [Emery Berger](https://emeryberger.com). It incorporates extensive feedback from many contributors.

This site was initially based on code and data collected by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) (UT-Austin), and the original faculty affiliation dataset was constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

This site uses information from [DBLP.org](http://dblp.org) which is made available under the ODC Attribution License.

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).