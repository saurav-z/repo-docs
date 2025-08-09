# CSrankings: The Premier Ranking of Top Computer Science Schools

**CSrankings provides a data-driven and objective ranking of computer science schools, based on faculty publications in top conferences, unlike rankings based on surveys.**  Access the code and data behind the ranking at the [original repository on GitHub](https://github.com/emeryberger/CSrankings).

## Key Features

*   **Metrics-Based Ranking:**  CSrankings uses a metrics-based approach, analyzing publications by faculty in the most selective computer science conferences.
*   **Objective & Data-Driven:**  The ranking avoids subjective survey-based methods, focusing on quantifiable research output to rank institutions.
*   **Difficult-to-Game Approach:**  The focus on highly selective conference publications is designed to be difficult to manipulate, providing a more reliable assessment of research excellence.
*   **Open Source:** All code and data are publicly available, allowing for transparency and community contributions.
*   **Quarterly Updates:**  The ranking is updated quarterly to reflect the latest research activity.

## Contributing

You can contribute to the CSrankings project by:

*   **Adding or Modifying Affiliations:**  Submit pull requests to update faculty affiliations.
*   Refer to the [`CONTRIBUTING.md`](CONTRIBUTING.md) file for full details on how to contribute.
*   **Shallow Clone for Quick Contributions:** Make contributions via a shallow clone, see instructions above.

## Running Locally

To run the CSrankings website locally, you will need to:

1.  Download DBLP data by running `make update-dblp`.
2.  Rebuild databases with `make`.
3.  Run a local web server (e.g., `python3 -m http.server`) and connect to [http://0.0.0.0:8000](http://0.0.0.0:8000).

You will also need to install the following dependencies: `libxml2-utils`, `npm`, `typescript`, `closure-compiler`, `python-lxml`, [pypy](https://doc.pypy.org/en/latest/install.html), and `basex`.

## Acknowledgements

This project was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). It builds upon the work of Swarat Chaudhuri, Papoutsaki et al., and utilizes data from [DBLP.org](http://dblp.org).

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).