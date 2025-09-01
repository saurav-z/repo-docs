# CS Rankings: A Data-Driven Ranking of Top Computer Science Schools

**Tired of rankings based on surveys?** CS Rankings provides a data-driven, metric-based approach to ranking computer science schools, focusing on faculty research publications in top conferences. 
Check out the original repository for more information: [https://github.com/emeryberger/CSrankings](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Metric-Based Rankings:** Utilizes the number of publications by faculty in the most selective computer science conferences.
*   **Avoids Manipulation:** Designed to be difficult to game, unlike citation-based or survey-based rankings.
*   **Open-Source:** All code and data used to build the website are available.
*   **Community-Driven:**  Contributions are welcome for adding or modifying affiliations.
*   **Comprehensive Data:** Leverages data from DBLP.org to provide a broad overview of research output.

## How CS Rankings Works

This ranking system focuses on the research output of computer science faculty. It measures the number of publications at the most selective conferences in various computer science areas. This method aims to provide a more objective and less easily manipulated ranking compared to traditional survey-based approaches.

## Contributing

### Adding or Modifying Affiliations

To contribute to the project, please:

1.  Review the `CONTRIBUTING.md` file for detailed instructions.
2.  All data is stored in `csrankings-[a-z].csv` files, with entries alphabetized by first name.
3.  Submit pull requests. Updates are processed quarterly.

## Running Locally

To run the site locally, you will need to:

1.  Download the DBLP data: ``make update-dblp`` (requires ~19GiB of memory).
2.  Rebuild the databases: ``make``.
3.  Run a local web server (e.g., ``python3 -m http.server``) and connect to [http://0.0.0.0:8000](http://0.0.0.0:8000).
4.  Install the necessary dependencies as listed in the original README.

### Quick Contribution with a Shallow Clone

For quick contributions without a full clone, use a shallow clone:

1.  Fork the CSrankings repository.
2.  Clone your fork with depth 1: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes on a branch, push, and create a pull request.

## Acknowledgements

Developed primarily by [Emery Berger](https://emeryberger.com) and based on work by Swarat Chaudhuri and Papoutsaki et al.  This project utilizes data from [DBLP.org](http://dblp.org).

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).