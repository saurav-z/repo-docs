# CSrankings: Your Definitive Guide to Top Computer Science Schools (and Repository)

**Tired of subjective university rankings?** CSrankings.org provides a data-driven, metrics-based ranking of computer science schools based on faculty research publications, offering a transparent and objective assessment. This repository contains the code and data that powers the CSrankings website. [View the live rankings at CSrankings.org](https://csrankings.org)

[View the original repository on GitHub](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Metrics-Driven Ranking:**  CSrankings utilizes a metrics-based approach, focusing on faculty publications in top-tier computer science conferences.
*   **Objective Assessment:**  The rankings are designed to be difficult to "game," relying on the difficulty of publishing in highly selective conferences rather than subjective surveys.
*   **Transparent Data:**  All data used to generate the rankings is available and open for community contribution, fostering transparency and accuracy.
*   **Regular Updates:**  The rankings are updated quarterly to ensure the most current assessment of faculty research activity.

## How to Contribute:

You can contribute to this project by:

*   **Adding or Modifying Affiliations:** Submit pull requests to update the `csrankings-[a-z].csv` files with new or corrected faculty affiliations. Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for full details on how to contribute. *Note: Updates are processed quarterly.*
*   **Running Locally:**  Follow the instructions to set up a local development environment and test changes before submitting a pull request.
*   **Shallow Cloning for Quick Contributions:**  Use a shallow clone to quickly contribute changes without a full local repository clone.

##  Technical Details:

*   **Data Source:**  DBLP.org, made available under the ODC Attribution License, is used for publication data.
*   **Development Stack:** Requires `libxml2-utils`, `npm`, `typescript`, `closure-compiler`, `python-lxml`, `pypy`, and `basex`.
*   **Make Commands:**  Use `make update-dblp` to download DBLP data and `make` to rebuild the databases.

## Acknowledgements:

This project was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). The project incorporates extensive feedback from the community and builds upon the work of [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) and the faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

## License:

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).