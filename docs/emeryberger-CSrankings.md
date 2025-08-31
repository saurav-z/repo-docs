# CS Rankings: Your Comprehensive Guide to Top Computer Science Schools

**CS Rankings offers a data-driven, metrics-based approach to ranking computer science schools, focusing on research output at top conferences.**  [Explore the original repository on GitHub](https://github.com/emeryberger/CSrankings).

## Key Features:

*   **Metrics-Based Ranking:** Unlike survey-based rankings, CS Rankings uses publication counts at highly selective computer science conferences to identify leading institutions and faculty.
*   **Data-Driven Methodology:**  The ranking is based on objective metrics, making it less susceptible to manipulation.
*   **Quarterly Updates:**  The rankings are updated quarterly to reflect the latest research output.
*   **Contribution Welcome:**  The project welcomes contributions to add or modify affiliations.  See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.
*   **Open Source and Transparent:** The code and data are available for anyone to use and inspect.

## How It Works

CS Rankings analyzes publications by faculty at the most selective computer science conferences. The goal is to provide a transparent and objective view of research activity. The data is sourced from DBLP.org.

## Contribute to the Project

You can contribute to the CS Rankings project. See the README for instructions on contributing.

## Getting Started

### Running Locally

To run the site locally, you'll need to:

1.  Download the DBLP data: `make update-dblp` (requires ~19GB memory).
2.  Build the databases: `make`
3.  Run a local web server (e.g., `python3 -m http.server`)
4.  Access the site: `http://0.0.0.0:8000`

### Required Dependencies
Install the necessary packages:
``apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler``

## Shallow Clone for Quick Contributions

For quick contributions without a full clone, use a shallow clone:

1.  Fork the CSrankings repo.
2.  Clone your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes on a branch, push them, and create a pull request.
## Acknowledgements
This site was developed primarily by and is maintained by [Emery Berger](https://emeryberger.com). It incorporates extensive feedback
from too many folks to mention here, including many contributors who
have helped to add and maintain faculty affiliations, home pages, and
so on.

This site was initially based on code and
data collected by [Swarat
Chaudhuri](https://www.cs.utexas.edu/~swarat/) (UT-Austin), though
it has evolved considerably since its inception. The
original faculty affiliation dataset was constructed by [Papoutsaki et
al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html); since
then, it has been extensively cleaned and updated by numerous
contributors. A previous ranking
also used DBLP and Brown's dataset for [ranking theoretical computer
science](https://projects.csail.mit.edu/dnd/ranking/.).

This site uses information from [DBLP.org](http://dblp.org) which is made
available under the ODC Attribution License.

## License

CSRankings is covered by the [Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International
License](https://creativecommons.org/licenses/by-nc-nd/4.0/).