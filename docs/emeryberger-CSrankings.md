# CSrankings: The Data-Driven Ranking of Top Computer Science Schools

**Tired of subjective rankings? CSrankings.org provides a metric-driven, objective assessment of computer science programs based on faculty publication records.** ([View the original repository](https://github.com/emeryberger/CSrankings))

## Key Features:

*   **Objective Ranking:** Unlike surveys, CSrankings.org uses a metrics-based approach, relying on faculty publications in top computer science conferences.
*   **Difficult-to-Game Methodology:**  The ranking focuses on publications in highly selective conferences, making it challenging to manipulate.
*   **Comprehensive Data:**  The ranking leverages data from DBLP.org to provide a broad view of research activity across various computer science areas.
*   **Community-Driven:**  Contributions are welcome.  You can submit pull requests to add or modify affiliations. (Note: Updates processed quarterly).
*   **Easy Contributions:**  Offers quick contribution via shallow cloning for those who do not want to clone the full repository.

## Contributing

You can contribute to the CSrankings project by:

*   Adding or modifying faculty affiliations.
    *   All data is in `csrankings-[a-z].csv` files.
    *   Read `CONTRIBUTING.md` for contribution details.
    *   Contributions processed quarterly.
*   Using a shallow clone to quickly make changes to the repository.

## Running Locally

To run the site locally, follow these steps:

1.  Download the DBLP data: ``make update-dblp`` (requires approximately 19GiB of memory).
2.  Rebuild the databases: ``make``
3.  Run a local web server (e.g., ``python3 -m http.server``)
4.  Access the site at [http://0.0.0.0:8000](http://0.0.0.0:8000).

**Dependencies:**  You will need to install several dependencies.  Check the original README for installation details.

## Acknowledgements

This project was developed and is maintained by [Emery Berger](https://emeryberger.com), with significant contributions from numerous individuals. It builds upon the work of Swarat Chaudhuri, Papoutsaki et al., and draws upon data from DBLP.org.

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).