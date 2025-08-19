# CSrankings: Data-Driven Rankings of Top Computer Science Schools

This repository powers CSrankings.org, a data-driven ranking system for computer science schools based on faculty publications at top conferences. [Visit the original repository on GitHub](https://github.com/emeryberger/CSrankings).

## Key Features

*   **Metrics-Based Approach:** Ranks institutions and faculty based on the number of publications in highly selective computer science conferences.
*   **Avoids Survey-Based Bias:** Unlike traditional rankings, this system relies entirely on objective metrics, minimizing the impact of subjective surveys.
*   **Difficult to Game:** The focus on publications in top conferences aims to provide a robust and reliable ranking system, resistant to manipulation.
*   **Open Source and Transparent:** The code and data used to generate the rankings are publicly available, promoting transparency and community contribution.
*   **Quarterly Updates:** The ranking data is updated on a quarterly basis to ensure relevance.

## Contributing

We welcome contributions to improve the accuracy and coverage of the rankings.  Follow these steps to contribute:

*   **Data Files:** Edit the `csrankings-[a-z].csv` files to add or modify faculty affiliations.
*   **Contribution Guidelines:** Please review `CONTRIBUTING.md` for detailed instructions on how to contribute.
*   **GitHub Direct Editing:** You can directly edit files in your GitHub fork and submit pull requests.
*   **Shallow Clone:** For quick contributions, use a shallow clone of your fork to bypass the full repository size.

## Running the Site Locally

To run the CSrankings website locally, you will need to set up the necessary dependencies and rebuild the databases.

1.  **Install Dependencies:**

    *   `apt-get install libxml2-utils npm python-lxml basex` (or equivalent for your system)
    *   `npm install -g typescript google-closure-compiler`
    *   Install `pypy`
2.  **Download DBLP Data:**  `make update-dblp` (requires ~19GB of memory)
3.  **Rebuild Databases:** `make`
4.  **Run Local Web Server:**  (e.g., `python3 -m http.server`)
5.  **Access the Site:** Open [http://0.0.0.0:8000](http://0.0.0.0:8000) in your browser.

## Acknowledgements

CSrankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). This project benefits from the contributions of many individuals and incorporates data from DBLP.org, made available under the ODC Attribution License.

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).