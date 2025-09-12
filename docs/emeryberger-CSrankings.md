# CSrankings: The Definitive Ranking of Top Computer Science Schools

**CSrankings provides a data-driven, metrics-based ranking of top computer science schools, offering a transparent and objective view of research performance.**  This ranking focuses on identifying institutions and faculty actively engaged in research across various computer science areas.

[View the original repository on GitHub](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Metrics-Based Approach:**  Rankings are based on the number of publications by faculty in top-tier computer science conferences, offering a robust and difficult-to-manipulate evaluation.
*   **Transparent Methodology:** Unlike rankings based on surveys, CSrankings uses a transparent, data-driven approach, focusing on objective research output.
*   **Comprehensive Data:**  Leverages the DBLP database to track publications and faculty affiliations across numerous computer science research areas.
*   **Community-Driven Updates:** Allows for community contributions to improve accuracy by submitting pull requests. (Updates are processed quarterly)
*   **Easy to Contribute:** You can now edit data files directly in GitHub to create pull requests
*   **Detailed Information:** Access the FAQ to learn more about the methodology and specific criteria used.

## Contributing

### Adding or Modifying Affiliations

*   **Contribution Process:**  Submit pull requests to update faculty affiliations.
*   **Data Files:** All data is stored in `csrankings-[a-z].csv` files.
*   **Contribution Guidelines:** See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

### Trying it Out at Home

To run the site locally, follow these steps:

1.  Download DBLP data: `make update-dblp` (requires ~19GiB of memory).
2.  Rebuild databases: `make`.
3.  Run a local web server (e.g., `python3 -m http.server`).
4.  Access the site at [http://0.0.0.0:8000](http://0.0.0.0:8000).

**Required Dependencies:** Install libxml2-utils, npm, typescript, closure-compiler, python-lxml, pypy and basex using the commands shown in the original README.

### Quick Contribution via a Shallow Clone

Use this approach to contribute changes without a full local clone:

1.  Fork the CSrankings repo.
2.  Do a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`.
3.  Make your changes, push them, and create a pull request.

## Acknowledgements

*   Developed and maintained by [Emery Berger](https://emeryberger.com).
*   Based on code and data from Swarat Chaudhuri and Papoutsaki et al.
*   Uses data from [DBLP.org](http://dblp.org) under the ODC Attribution License.

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).