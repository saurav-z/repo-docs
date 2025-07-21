# CS Rankings: The Definitive Ranking of Top Computer Science Schools

**CS Rankings** provides a data-driven, metric-based ranking of computer science institutions and faculty, highlighting research productivity across various CS disciplines. ([Original Repository](https://github.com/emeryberger/CSrankings))

## Key Features:

*   **Metrics-Based:** Rankings are determined by the number of publications by faculty in top-tier computer science conferences, offering a data-driven approach.
*   **Difficult to Game:** The focus on highly selective conference publications aims to provide a robust ranking, less susceptible to manipulation compared to citation-based metrics.
*   **Regular Updates:** The rankings are updated quarterly, reflecting the dynamic nature of research output.
*   **Community-Driven:** Contributions from the community are welcomed for adding/modifying affiliations and other data improvements (see [CONTRIBUTING.md](CONTRIBUTING.md)).
*   **Open Data:** Utilizes data from DBLP.org, available under the ODC Attribution License.

## Contributing

### Adding or Modifying Affiliations

Updates are processed quarterly.  You can submit pull requests at any time.

*   All data is located in `csrankings-[a-z].csv` files, organized by the initial letter of the author's first name.
*   See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

### Shallow Clone for Quick Contributions

Avoid a full clone with these steps:

1.  Fork the CSrankings repository.
2.  Do a shallow clone: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes on a branch, push, and create a pull request.

## Getting Started Locally

To run the site locally:

1.  Download the DBLP data: `make update-dblp` (requires ~19GB memory).
2.  Rebuild the databases: `make`
3.  Start a local web server: `python3 -m http.server`
4.  Access the site: `http://0.0.0.0:8000`

### Required Dependencies:

You'll need to install the following:

```bash
apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
```

You may also need [pypy](https://doc.pypy.org/en/latest/install.html).

## Acknowledgements

Developed and maintained by [Emery Berger](https://emeryberger.com), with contributions from many community members.  Based on the work of [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/), and the original faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).