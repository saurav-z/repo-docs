# CS Rankings: A Metrics-Based Ranking of Top Computer Science Schools

This repository powers the CS Rankings website, a data-driven resource that ranks computer science schools based on the research productivity of their faculty. Access the live rankings at [https://csrankings.org](https://csrankings.org) and explore the code behind it here!  ([Original Repo](https://github.com/emeryberger/CSrankings))

## Key Features

*   **Metrics-Driven Ranking:** Unlike rankings based on surveys, CS Rankings uses a metrics-based approach, measuring publications in top computer science conferences.
*   **Difficult-to-Game Methodology:** This approach is designed to be robust and less susceptible to manipulation compared to methods relying on citations or surveys.
*   **Comprehensive Data:** The rankings leverage data from DBLP and a curated dataset of faculty affiliations.
*   **Community Contribution:**  The project welcomes contributions to update faculty affiliations and improve the dataset.

## Contributing

Contributions are welcome! Please follow these guidelines:

*   **Data Files:**  Faculty affiliation data is stored in `csrankings-[a-z].csv` files, alphabetized by first name initial.
*   **Contribution Process:** Edit files directly in GitHub to create pull requests. Updates are processed quarterly.
*   **CONTRIBUTING.md:** Refer to the `CONTRIBUTING.md` file for detailed instructions.
*   **Shallow Clone for Quick Contributions:**  Use a shallow clone to contribute without downloading the full repository history.

## Running Locally

To run CS Rankings locally:

1.  Download DBLP data: `make update-dblp` (requires ~19GiB memory).
2.  Rebuild databases: `make`.
3.  Run a local web server (e.g., `python3 -m http.server`) and connect to [http://0.0.0.0:8000](http://0.0.0.0:8000).
4.  Install dependencies: `apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler`

## Acknowledgements

This project was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). It builds upon initial work by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) and a faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html). The project also utilizes information from [DBLP.org](http://dblp.org).

## License

CS Rankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).