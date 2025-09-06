# CS Rankings: The Premier Data-Driven Ranking of Computer Science Schools

**CS Rankings provides a comprehensive, objective, and data-driven ranking of computer science institutions worldwide, helping prospective students, faculty, and industry professionals make informed decisions.**

[Explore the CS Rankings Website](https://csrankings.org/)

This repository contains all the code and data used to build the CS Rankings website. Unlike rankings based on surveys, CS Rankings utilizes a metrics-based approach, focusing on faculty publications in top-tier computer science conferences to provide a robust and reliable assessment of research excellence.

**Key Features:**

*   **Data-Driven Methodology:** Rankings are based on publication records in highly selective computer science conferences, avoiding subjective survey-based approaches.
*   **Objective Assessment:** Provides an unbiased evaluation of research productivity across various computer science areas.
*   **Up-to-Date Data:** Continuously updated to reflect the latest research output from institutions and faculty.
*   **Community Driven:** Relies on contributions from the community for data maintenance and additions.
*   **Easy to Contribute:** The repository is open for contributions to the data and website.

## How CS Rankings Works

CS Rankings measures the research output of computer science faculty by counting their publications at the most selective conferences in each area of computer science. This methodology is designed to be resistant to manipulation and provides a clear indication of research activity and impact.

## Contributing

We welcome contributions to enhance the CS Rankings dataset and website.

*   **Adding or Modifying Affiliations:** Follow the instructions in `CONTRIBUTING.md` to add or update faculty affiliations. Note that updates are processed quarterly.
*   **Making Changes:** You can create pull requests with changes directly within GitHub or by cloning the repository locally.
*   **Shallow Cloning:** To contribute without cloning the entire repository, use a shallow clone to download only the most recent commit.
*   All data is in the files `csrankings-[a-z].csv`, with authors listed in alphabetical order by their first name, organized by the initial letter. Please read <a
href="CONTRIBUTING.md">```CONTRIBUTING.md```</a> for full details on
how to contribute.

## Running the Site Locally

To run the CS Rankings website locally, you will need to:

1.  Download the DBLP data: ``make update-dblp``
2.  Rebuild the databases: ``make``
3.  Run a local web server (e.g., ``python3 -m http.server``)
4.  Connect to [http://0.0.0.0:8000](http://0.0.0.0:8000)

**Dependencies:**

You will need to install the following dependencies: `libxml2-utils`, `npm`, `typescript`, `closure-compiler`, `python-lxml`, `pypy`, and `basex`.

Install dependencies by running a command like:
``apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler``

## Acknowledgements

CS Rankings was developed primarily by [Emery Berger](https://emeryberger.com) and incorporates feedback from a wide community of contributors.  It is based on code and data from Swarat Chaudhuri and the original faculty affiliation dataset constructed by Papoutsaki et al. This site uses information from [DBLP.org](http://dblp.org) which is made available under the ODC Attribution License.

## License

CS Rankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

## Original Repository

[View the original repository on GitHub](https://github.com/emeryberger/CSrankings)