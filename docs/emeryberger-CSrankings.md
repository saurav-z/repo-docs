# CSrankings: A Metrics-Based Ranking of Top Computer Science Schools

**CSrankings.org** provides a data-driven, objective ranking of computer science departments worldwide, based on faculty publications at top-tier conferences.  Explore the original repository on [GitHub](https://github.com/emeryberger/CSrankings) for more details.

## Key Features:

*   **Metrics-Based Ranking:** Unlike rankings based on surveys, CSrankings relies entirely on publication data from highly selective computer science conferences.
*   **Difficult-to-Game Approach:** This methodology aims to create a ranking that is less susceptible to manipulation compared to citation-based metrics.
*   **Focus on Active Research:** Identifies institutions and faculty actively engaged in research across various computer science fields.
*   **Regular Updates:** The rankings are updated on a quarterly basis to ensure accuracy and relevance.

## Contributing

### Adding or Modifying Affiliations

*   **Quarterly Updates:**  Changes are processed quarterly. Submit pull requests anytime; they will be integrated during the next update cycle.
*   **Direct Editing:** Utilize GitHub's interface to edit the `csrankings-[a-z].csv` files.
*   **Contribution Guidelines:**  Refer to the `CONTRIBUTING.md` file for detailed instructions on how to contribute.

### Running CSrankings Locally

1.  **Data Download:**  Download the DBLP data:  `make update-dblp` (requires ~19GB of memory).
2.  **Build Databases:** Run `make` to rebuild the databases.
3.  **Local Server:** Start a local web server (e.g., `python3 -m http.server`) and access the site at [http://0.0.0.0:8000](http://0.0.0.0:8000).
4.  **Required Dependencies:** Ensure you have the necessary tools and libraries installed: `libxml2-utils`, `npm`, `typescript`, `google-closure-compiler`, `python-lxml`, `pypy`, and `basex`.  Use the command: `apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler`

### Quick Contribution via a Shallow Clone

For smaller contributions, avoid a full clone with these steps:

1.  **Fork:**  Fork the CSrankings repository on GitHub.
2.  **Shallow Clone:**  Clone your fork with: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  **Create Pull Request:**  Make changes, push to your clone, and create a pull request.

## Acknowledgements

CSrankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com).  It builds upon initial work by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) and utilizes data from DBLP.org (under the ODC Attribution License).  The original faculty affiliation dataset was constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).