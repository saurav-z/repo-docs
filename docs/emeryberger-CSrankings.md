# CSrankings: Your Comprehensive Guide to Top Computer Science Schools

Looking for the best computer science programs? **CSrankings.org provides a metrics-based ranking of computer science schools, identifying institutions and faculty actively engaged in research across various areas of computer science.** This approach offers a unique perspective by focusing on publications in highly selective conferences, aiming to provide a more robust and less easily manipulated evaluation than methods based on surveys or citations alone. 

[Visit the original repository on GitHub](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Metrics-Based Ranking:**  Evaluates schools based on the number of publications by faculty in the most selective computer science conferences.
*   **Data-Driven Approach:**  Employs a data-driven methodology, avoiding the subjectivity of survey-based rankings.
*   **Areas of Computer Science:** Offers rankings across a variety of computer science areas.
*   **Difficult-to-Game Methodology:** Designed to be resistant to manipulation compared to citation-based metrics.
*   **Regular Updates:** The site is updated quarterly to reflect the latest research activity.

## Contributing

### Adding or modifying affiliations
*   Updates are processed quarterly.
*   Submit pull requests via the `csrankings-[a-z].csv` files.
*   See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Shallow Clone for Quick Contributions

1.  Fork the CSrankings repo.
2.  Do a shallow clone: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes on a branch and create a pull request.

## Requirements to Run Locally
To run the site locally, you'll need to download the DBLP data (`make update-dblp`) and then rebuild the databases (`make`). You will need to install several dependencies. See the original README for the exact list of dependencies.

## Acknowledgements

This project was primarily developed and is maintained by [Emery Berger](https://emeryberger.com).
It incorporates data and code from:

*   [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/)
*   [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html)
*   [DBLP.org](http://dblp.org)

## License

CSRankings is covered by the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).