# CSrankings: A Data-Driven Ranking of Top Computer Science Schools

This repository houses the code and data that power CSrankings.org, a website that provides a metrics-based ranking of computer science schools, focusing on research productivity. [Explore the live rankings here](https://csrankings.org/) or dive into the source code below!  

[View the original repository on GitHub](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Metrics-Based Ranking:** Rankings are derived from the number of publications by faculty at top computer science conferences, providing a data-driven approach.
*   **Avoids Gaming:** Unlike rankings based on surveys or citations, this method focuses on publications in highly selective conferences, making it harder to manipulate.
*   **Comprehensive Data:** The ranking utilizes extensive data from DBLP and other sources, reflecting a broad view of computer science research.
*   **Community Driven:** Contributions from the community are welcome, with clear guidelines for adding or modifying affiliations.

## Contributing

### Adding or Modifying Affiliations

**_NOTE: Updates are now processed on a quarterly basis. You may submit pull requests at any time, but they may not be processed until the next quarter (after three months have elapsed)._**

You can now edit files directly in GitHub to create pull requests. All data is in the files `csrankings-[a-z].csv`, with authors listed in alphabetical order by their first name, organized by the initial letter. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for full details on how to contribute.

## Running Locally

To run CSrankings locally, follow these steps:

1.  **Download DBLP Data:**  Run `make update-dblp` (requires ~19GiB memory).
2.  **Build Databases:** Run `make`.
3.  **Test:** Run a local web server (e.g., `python3 -m http.server`) and connect to `http://0.0.0.0:8000`.

### Dependencies

You will also need to install the following dependencies:

*   libxml2-utils (or equivalent)
*   npm
*   typescript
*   google-closure-compiler
*   python-lxml
*   [pypy](https://doc.pypy.org/en/latest/install.html)
*   basex

Use the following command (or equivalent for your distribution):

```bash
apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
```

## Quick Contribution with a Shallow Clone

To contribute without a full clone:

1.  Fork the CSrankings repo.
2.  Shallow clone your fork: `git clone --depth 1 https://github.com/yourusername/CSrankings`
3.  Make changes, push to your clone, and create a pull request.

## Acknowledgements

This site was developed primarily by and is maintained by [Emery Berger](https://emeryberger.com). It incorporates extensive feedback from many contributors.

This site was initially based on code and data collected by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) (UT-Austin). The original faculty affiliation dataset was constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html).

This site uses information from [DBLP.org](http://dblp.org) which is made available under the ODC Attribution License.

## License

CSRankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).