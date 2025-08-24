# CSrankings: The Definitive, Metrics-Based Ranking of Top Computer Science Schools

**Tired of rankings based on surveys? CSrankings provides an objective, metrics-driven assessment of computer science programs worldwide, focusing on research output.**  This repository houses the code and data behind the popular CSrankings website.

[View the live CSrankings website](https://csrankings.org/) | [Explore the Original Repository on GitHub](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Objective Ranking:**  Based entirely on metrics, specifically the number of publications by faculty at top computer science conferences.
*   **Data-Driven Approach:** Leverages publication data, offering a more reliable assessment than survey-based methods.
*   **Difficulty to Game:** Designed to be difficult to manipulate, unlike rankings that rely on citations.
*   **Comprehensive Data:**  Utilizes data from DBLP.org, ensuring a wide-ranging and up-to-date dataset.
*   **Open Source:** The code and data are available for public review and contribution.

## Contributing

This project is open to contributions!  You can contribute by:

*   **Adding or modifying affiliations:**  Data is stored in `csrankings-[a-z].csv` files.  Follow the guidelines in `CONTRIBUTING.md` for detailed instructions.
*   **Submitting pull requests:**  Contributions are processed quarterly, so your changes will be reflected in the next update.
*   **Shallow Clone for Quick Contribution:**  For smaller contributions, utilize a shallow clone of your fork.

## Getting Started Locally

To run the site locally:

1.  **Install Dependencies:** Install the required tools, including `libxml2-utils`, `npm`, `typescript`, `closure-compiler`, `python-lxml`, [pypy](https://doc.pypy.org/en/latest/install.html), and `basex`.
2.  **Download DBLP Data:** Run `make update-dblp` (requires ~19GB memory).
3.  **Build Databases:** Run `make`.
4.  **Test Locally:** Run a local web server (e.g., `python3 -m http.server`) and access the site at `http://0.0.0.0:8000`.

## Acknowledgements

CSrankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com).  The project builds upon the work of:

*   [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) (UT-Austin)
*   [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html) (for the original faculty affiliation dataset)
*   Many other contributors.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).