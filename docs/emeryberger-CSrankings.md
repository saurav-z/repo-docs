# CSrankings: The Premier Ranking of Top Computer Science Schools

**Looking for the best computer science programs?** CSrankings provides a data-driven, metrics-based ranking of computer science schools, focusing on research productivity at top conferences.  This ranking is designed to be difficult to manipulate, unlike rankings based on surveys or citations.  Explore the methodology and data to see how schools are assessed.

[Link to the original repository](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Metrics-Driven Ranking:**  Based on publications by faculty at the most selective computer science conferences.
*   **Objective Assessment:** Unlike surveys, this ranking is based entirely on metrics, aiming to avoid manipulation.
*   **Detailed Data:** The repository contains all code and data used to build the website.
*   **Community Driven:** Includes contributions to add and maintain faculty affiliations, home pages, etc.
*   **Regular Updates:** The ranking is updated quarterly.

## How CSrankings Works:

The ranking utilizes a metrics-based approach, focusing on the number of publications by faculty at the most selective conferences in each computer science area. This methodology emphasizes research productivity and aims to provide a reliable assessment of institutions and faculty actively involved in computer science research.

## Contributing to CSrankings:

Contributions are welcome! You can submit changes to faculty affiliations and other data.

*   **Contribution Process:** Updates are processed quarterly. Submit pull requests anytime; they will be reviewed in the next cycle.
*   **Data Files:** Data is stored in `csrankings-[a-z].csv` files, with authors alphabetized by their first name.
*   **Shallow Clone:** To contribute without cloning the full repository, use a shallow clone.
*   **Full details:** Read the `CONTRIBUTING.md` for more information.

## Running CSrankings Locally:

To run the site locally, you'll need to:

1.  Download DBLP data (`make update-dblp`)
2.  Rebuild databases (`make`).
3.  Set up a local web server (e.g., `python3 -m http.server`) and view at `http://0.0.0.0:8000`.

## Dependencies:

You will need to install several dependencies, including:

```bash
apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
```

You'll also need to install [pypy](https://doc.pypy.org/en/latest/install.html).

## Acknowledgements:

CSrankings was developed primarily by [Emery Berger](https://emeryberger.com). It is based on work by Swarat Chaudhuri and the faculty affiliation dataset constructed by Papoutsaki et al. DBLP.org provides valuable information to this site.

## License:

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).