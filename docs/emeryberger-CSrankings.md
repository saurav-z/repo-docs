# CS Rankings: Your Definitive Guide to Top Computer Science Schools

**Uncover the leading computer science institutions and faculty based on rigorous, metrics-driven research performance.** ([View the Original Repo](https://github.com/emeryberger/CSrankings))

This ranking provides a data-driven assessment of computer science departments, focusing on publication output at premier conferences.  Unlike rankings based on surveys, CS Rankings offers a transparent and objective view of research excellence.

**Key Features:**

*   **Metrics-Based:** Rankings are determined by the number of publications from faculty at highly selective computer science conferences.
*   **Difficult to Game:** Avoids manipulation common in citation-based metrics, ensuring a more reliable assessment of research quality.
*   **Comprehensive:** Provides a broad overview of research performance across various computer science areas.
*   **Community-Driven:** Contributions from the community help maintain up-to-date faculty affiliations.

## How CS Rankings Works

CS Rankings uses a data-driven approach to evaluate computer science departments and faculty. It assesses research output based on publications at the most selective conferences in computer science. The ranking methodology aims to be robust and difficult to manipulate.

## Contributing to CS Rankings

The CS Rankings project is community-driven. Contribute to the project by:

*   **Adding or Modifying Affiliations:** Follow the guidelines in `CONTRIBUTING.md` to submit pull requests with updates.  Note that updates are processed on a quarterly basis.
*   **GitHub Direct Editing:** Make quick changes directly within GitHub.
*   **Shallow Clone for Contributions:** Use a shallow clone of the repository for efficient contributions without a full local clone.

## Setting Up and Running Locally

To set up and run the site locally, follow these steps:

1.  **Download DBLP Data:** Run `make update-dblp` (requires ~19GB memory).
2.  **Build Databases:** Run `make`.
3.  **Test Locally:** Use a local web server (e.g., `python3 -m http.server`) and access at `http://0.0.0.0:8000`.
4.  **Install Dependencies:** Ensure required packages, including libxml2-utils, npm, typescript, closure-compiler, python-lxml, pypy, and basex, are installed.

## Acknowledgements

CS Rankings was developed primarily by [Emery Berger](https://emeryberger.com). This site is inspired by and incorporates data from projects including Swarat Chaudhuri, the faculty affiliation dataset constructed by Papoutsaki et al., and DBLP.org.

## License

CS Rankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).