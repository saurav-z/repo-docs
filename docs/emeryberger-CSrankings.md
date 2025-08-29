# CSrankings: The Premier Computer Science School Ranking

**CSrankings provides a data-driven, objective ranking of top computer science institutions and faculty based on research output.** Explore the latest rankings, discover leading researchers, and gain insights into the dynamic landscape of computer science.  [View the original repository here](https://github.com/emeryberger/CSrankings).

## Key Features

*   **Metrics-Driven Rankings:** CSrankings uses a metrics-based approach, measuring the number of publications by faculty at top computer science conferences.
*   **Objective and Data-Based:** Unlike rankings based on surveys, CSrankings relies entirely on publicly available data, making it difficult to manipulate and game.
*   **Focus on Research Excellence:** The ranking methodology prioritizes publications in highly selective conferences, reflecting active engagement in research across various computer science areas.
*   **Regular Updates:** The rankings are updated quarterly to reflect the latest research output and changes in faculty affiliations.
*   **Transparent Methodology:** The approach is designed to be difficult to game and is detailed in the FAQ.

## Contributing

CSrankings welcomes contributions to enhance the accuracy and coverage of its data.

*   **Adding/Modifying Affiliations:** You can contribute by editing the `csrankings-[a-z].csv` files. Follow the guidelines in `CONTRIBUTING.md`. Note that updates are processed quarterly.
*   **Shallow Clone for Easy Contributions:** Use a shallow clone to contribute without downloading the full repository history, useful for quickly adding a faculty or affiliation.

## Setting Up Locally

To run CSrankings locally, you'll need:

*   **DBLP Data:** Download the DBLP data using `make update-dblp`.
*   **Build and Test:** Build the databases using `make`.
*   **Web Server:** Run a local web server (e.g., `python3 -m http.server`) and access it via `http://0.0.0.0:8000`.
*   **Dependencies:** Install the necessary packages, including `libxml2-utils`, `npm`, `typescript`, `google-closure-compiler`, `python-lxml`, `pypy`, and `basex`.

## Acknowledgements

CSrankings was developed primarily by [Emery Berger](https://emeryberger.com). The project draws on data and inspiration from various sources, including:

*   Swarat Chaudhuri (UT-Austin)
*   Papoutsaki et al. (Brown University)
*   DBLP.org (under the ODC Attribution License)

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).