# CS Rankings: Your Definitive Guide to Top Computer Science Schools

**Discover and compare the world's leading computer science institutions based on a rigorous, metrics-driven methodology.** (See the original repository: [https://github.com/emeryberger/CSrankings](https://github.com/emeryberger/CSrankings))

## Key Features:

*   **Metrics-Based Ranking:**  CS Rankings provides an objective evaluation of computer science departments, relying on the number of publications by faculty at top-tier conferences.
*   **Avoids Survey Reliance:** Unlike rankings based on subjective surveys, CS Rankings focuses on tangible research output, making it harder to game.
*   **Open Source & Community Driven:** The project is built with publicly available data and welcomes contributions from the community.
*   **Regular Updates:** The rankings are updated quarterly to reflect the latest research achievements.
*   **Detailed Data Source:** Uses data from DBLP.org, a comprehensive database of computer science publications.

## Contributing & Running Locally

### Adding or Modifying Affiliations

**_NOTE: Updates are now processed on a quarterly basis._**

You can contribute by editing the data files directly in GitHub and submitting a pull request. See `CONTRIBUTING.md` for detailed instructions.  Files are organized in `csrankings-[a-z].csv` by faculty last name.

### Running Locally

To run the site locally, you'll need to:

1.  Download DBLP data: `make update-dblp` (requires ~19GB memory).
2.  Rebuild databases: `make`.
3.  Serve the site: `python3 -m http.server` and access it at `http://0.0.0.0:8000`.

**Prerequisites:**

You'll need several dependencies installed, including:

*   `libxml2-utils` (or equivalent)
*   `npm`
*   `typescript`
*   `closure-compiler`
*   `python-lxml`
*   `pypy`
*   `basex`

Install them via a command like:

```bash
apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
```

### Quick Contribution via a Shallow Clone

To make changes without a full clone:

1.  Fork the repository.
2.  Create a shallow clone of your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes, commit to a branch, push, and create a pull request.

## Acknowledgements

CS Rankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com).  The project builds upon the work of [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/), [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html), and utilizes data from [DBLP.org](http://dblp.org) (ODC Attribution License).

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).