# CSrankings: Your Go-To Ranking of Top Computer Science Schools

**CSrankings is a data-driven, metrics-based ranking of computer science institutions and faculty based on research publication performance.**

[View the CSrankings website](https://csrankings.org)

This repository provides the code and data used to build the CSrankings website, offering a unique perspective on top computer science programs. Unlike rankings based on surveys, CSrankings uses a metrics-driven approach focusing on publications in top-tier computer science conferences.

**Key Features:**

*   **Metrics-Based Ranking:** Utilizes publications in selective computer science conferences.
*   **Data-Driven Approach:**  Provides an objective assessment of research output.
*   **Transparent Methodology:**  Openly available code and data for review and contribution.
*   **Community-Driven:** Contributions from the community to update and maintain the data.

**Contribute & Get Involved**

Help improve the CSrankings dataset!

*   **Adding or Modifying Affiliations:** Submit pull requests to the `csrankings-[a-z].csv` files. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
*   **Shallow Clone Contribution:** Contribute without a full clone using a shallow clone, ideal for quick updates.
*   **Quarterly Updates:** Updates are processed on a quarterly basis.

**Getting Started Locally**

To run the site locally:

1.  Download DBLP data: `make update-dblp` (requires ~19GB memory)
2.  Rebuild databases: `make`
3.  Test locally: Run a local web server (e.g., `python3 -m http.server`) and connect to [http://0.0.0.0:8000](http://0.0.0.0:8000).
4.  Install dependencies:

    ```bash
    apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
    ```
    or the equivalent for your distribution, including pypy.

**Acknowledgements**

Developed and maintained primarily by [Emery Berger](https://emeryberger.com), incorporating feedback from the community.  Based on initial code and data by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) and the faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

**License**

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

---
[Back to the original repository on GitHub](https://github.com/emeryberger/CSrankings)