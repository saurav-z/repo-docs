# CSrankings: The Definitive Ranking of Top Computer Science Schools

Tired of rankings based solely on surveys? CSrankings provides a **metrics-based and data-driven ranking** of computer science departments, using publication records in top conferences.

[Visit the original repository on GitHub](https://github.com/emeryberger/CSrankings)

**Key Features:**

*   **Metrics-Based Ranking:** Leverages publication counts in highly selective computer science conferences.
*   **Objective and Data-Driven:** Avoids subjective survey-based methodologies, focusing on quantifiable research output.
*   **Difficult to Game:** Designed to be resistant to manipulation, unlike citation-based approaches.
*   **Comprehensive Data:**  Based on data from DBLP and updated on a quarterly basis.
*   **Community Driven:** Open-source with contributions from the computer science community.

**How CSrankings Works:**

CSrankings evaluates institutions and faculty based on their publication records in leading computer science conferences. This approach prioritizes research output over subjective assessments, offering a transparent and objective view of research performance.

**Contributing:**

Help improve the accuracy of CSrankings!  You can contribute by:

*   Adding or modifying faculty affiliations (updates processed quarterly)
*   Submitting pull requests with your updates directly via GitHub (after reviewing the [CONTRIBUTING.md](CONTRIBUTING.md) guidelines)

**Getting Started (for local setup):**

1.  Clone the repository and download the DBLP data: `make update-dblp`
2.  Build the databases: `make`
3.  Run a local web server (e.g., `python3 -m http.server`) and access the site at `http://0.0.0.0:8000`.

   **Dependencies:**  You'll also need to install the required dependencies, as detailed in the original README.

**Acknowledgments:**

CSrankings was developed primarily by [Emery Berger](https://emeryberger.com). It builds upon the work of Swarat Chaudhuri (UT-Austin), Papoutsaki et al., and utilizes data from DBLP.org (ODC Attribution License).

**License:**

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).