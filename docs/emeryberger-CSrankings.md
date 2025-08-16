# CSrankings: A Metrics-Driven Ranking of Top Computer Science Schools

**CSrankings provides a data-driven, objective ranking of computer science institutions and faculty based on publications in top-tier conferences.** This repository contains the code and data that powers the CSrankings website.

[View the live website: https://csrankings.org](https://csrankings.org)

[Go to the original repository on GitHub](https://github.com/emeryberger/CSrankings)

**Key Features:**

*   **Metrics-Based Ranking:**  Uses publication counts in highly selective computer science conferences, providing a robust and objective ranking system.
*   **Difficult-to-Game Methodology:**  Employs a methodology designed to be resistant to manipulation, unlike survey-based or citation-based rankings.
*   **Open and Transparent:**  All code and data are publicly available, allowing for community contributions and transparency.
*   **Regularly Updated:** The ranking is updated quarterly to reflect the latest research output.

## Contributing

### Adding or Modifying Affiliations

**_NOTE: Updates are now processed on a quarterly basis. You may submit pull requests at any time, but they may not be processed until the next quarter (after three months have elapsed)._**

You can contribute by modifying the data files (`csrankings-[a-z].csv`) which are organized alphabetically by the first name of the author.

*   **Direct GitHub Edits:** Edit files directly within GitHub to create pull requests.
*   **Contribution Guidelines:** Please read the [`CONTRIBUTING.md`](CONTRIBUTING.md) file for detailed instructions on how to contribute.
*   **Shallow Clone Option:** For faster contribution without a full clone, use a shallow clone to make changes and create pull requests.

## Setting up Locally

### Prerequisites

You will need the following tools installed on your system:

*   libxml2-utils (or equivalent)
*   npm
*   typescript
*   google-closure-compiler
*   python-lxml
*   pypy
*   basex

Install these using:
```bash
apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
```

### Build and Run

1.  Download the DBLP data by running: ``make update-dblp`` (requires ~19GiB of memory).
2.  Rebuild the databases by running: ``make``.
3.  Test the website by running a local web server (e.g., ``python3 -m http.server``) and then connecting to [http://0.0.0.0:8000](http://0.0.0.0:8000).

## Acknowledgements

This project was developed by [Emery Berger](https://emeryberger.com) and incorporates contributions from numerous individuals. It is based on the work of Swarat Chaudhuri, Papoutsaki et al., and utilizes data from DBLP.org, which is available under the ODC Attribution License.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).