# CSrankings: Data-Driven Ranking of Top Computer Science Schools

**CSrankings provides an objective, metrics-based ranking of computer science departments worldwide, based on the publications of their faculty.** Find the source code and contribute to this valuable resource on [GitHub](https://github.com/emeryberger/CSrankings).

## Key Features:

*   **Metrics-Driven Approach:**  Uses publication data from selective computer science conferences to assess research productivity and impact, avoiding subjective survey-based rankings.
*   **Objective Ranking:**  Focuses on measurable research output, making it harder to manipulate rankings.
*   **Open-Source & Collaborative:**  Built on open-source code and data, and welcomes contributions from the community to improve accuracy and expand coverage.
*   **Regularly Updated:**  Data is processed and updated quarterly to reflect the latest research activity.
*   **Detailed Methodology:**  Provides transparency in its ranking methodology, with an FAQ for more details.
*   **Focus on Publications:** Based on publications, rather than citations, which can be easily manipulated.

## Contribute to CSrankings

**Want to contribute to this amazing resource?**
You can contribute by adding or modifying affiliations.

**Here's how to get involved:**

*   **Submit Pull Requests:** Changes are processed quarterly.  Submit pull requests at any time.
*   **Edit Directly on GitHub:**  Edit data files (`csrankings-[a-z].csv`) directly in GitHub to create pull requests.
*   **See CONTRIBUTING.md for details:** Read [CONTRIBUTING.md](CONTRIBUTING.md) for full instructions on how to contribute.
*   **Use a Shallow Clone:** Contribute without needing to clone the full repository using the shallow clone instructions.

## Running CSrankings Locally

**To run CSrankings locally, you'll need to:**

1.  Download the DBLP data using `make update-dblp` (requires ~19GB of memory).
2.  Rebuild the databases with `make`.
3.  Set up a local web server (e.g., `python3 -m http.server`) and access it through `http://0.0.0.0:8000`.

**Required Tools:**

*   libxml2-utils (or package with xmllint)
*   npm
*   typescript
*   google-closure-compiler
*   python-lxml
*   [pypy](https://doc.pypy.org/en/latest/install.html)
*   basex

Install these with a command like:

`apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler`

## Acknowledgements

CSrankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). The project incorporates data and feedback from numerous contributors.  It builds upon work by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) and the faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html). Data from [DBLP.org](http://dblp.org) is used under the ODC Attribution License.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).