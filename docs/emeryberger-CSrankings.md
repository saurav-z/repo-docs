# CSrankings: The Data-Driven Ranking of Top Computer Science Schools

**Tired of subjective university rankings?** CSrankings.org offers a metrics-based ranking of top computer science schools based on faculty publications in leading CS conferences, providing a transparent and objective assessment.

[Visit the Original Repository on GitHub](https://github.com/emeryberger/CSrankings)

## Key Features

*   **Metrics-Based Ranking:** Relies on publication counts in highly selective computer science conferences, avoiding surveys and subjective assessments.
*   **Transparent Methodology:**  Provides a clear, data-driven approach to ranking, minimizing manipulation.
*   **Up-to-Date Data:**  Regularly updated with the latest publication data.
*   **Community-Driven:**  Open for contributions, allowing the community to add and update faculty affiliations.
*   **Based on DBLP:**  Leverages the comprehensive DBLP database for publication data.

## How It Works

CSrankings.org uses a unique methodology to rank institutions and faculty actively engaged in computer science research. The ranking algorithm focuses on the number of publications by faculty that have appeared at the most selective conferences in each area of computer science.

## Contributing

Help improve CSrankings!  You can contribute by:

*   **Adding or Modifying Affiliations:**  Submit pull requests with changes to the `csrankings-[a-z].csv` files.  See the `CONTRIBUTING.md` file for detailed instructions.  **Note:** Updates are processed quarterly.
*   **Shallow Cloning:**  Contribute changes without a full repository clone via shallow clone (see details in README).

## Getting Started (Locally)

To run CSrankings locally, follow these steps:

1.  **Download DBLP Data:** Run `make update-dblp`. This requires approximately 19GB of memory.
2.  **Build Databases:**  Run `make`.
3.  **Run a Local Web Server:**  Use `python3 -m http.server` and connect to [http://0.0.0.0:8000](http://0.0.0.0:8000).
4.  **Install Dependencies:** Install required tools: `libxml2-utils`, `npm`, `typescript`, `closure-compiler`, `python-lxml`, `pypy`, and `basex`.
    *   Example: `apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler`

## Acknowledgements

CSrankings.org was developed primarily by [Emery Berger](https://emeryberger.com) and is based on the code and data originally collected by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/). The site also incorporates data from [DBLP.org](http://dblp.org) (ODC Attribution License) and the faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

## License

CSrankings is covered by the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).