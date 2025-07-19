# CS Rankings: The Definitive Ranking of Computer Science Schools

**CS Rankings** provides a comprehensive, data-driven ranking of top computer science schools worldwide, using a metrics-based approach to assess research productivity. Check out the original repository on GitHub! [https://github.com/emeryberger/CSrankings](https://github.com/emeryberger/CSrankings)

## Key Features

*   **Metrics-Based Ranking:** Ranks institutions based on the number of publications by faculty in the most selective computer science conferences.
*   **Difficult to Game:** Uses publication counts in top conferences, making it challenging to manipulate rankings.
*   **Data-Driven:** Relies on objective data, avoiding subjective survey-based methodologies.
*   **Open Source:** The code and data are available for community contributions and improvements.
*   **Regular Updates:** Rankings are updated quarterly to reflect the latest research output.

## How CS Rankings Works

CS Rankings takes a unique approach to ranking computer science departments, focusing on research output. Unlike rankings that rely on surveys, this site uses a purely metrics-based approach. It measures the number of publications by faculty at top computer science schools that have appeared at the most selective conferences in each area of computer science.

This approach is designed to be difficult to game, as publishing in such conferences is generally difficult. It aims to provide a fair and transparent way to assess the research productivity of computer science institutions.

## Contributing

Contributions are welcome! You can submit pull requests to add or modify faculty affiliations.

*   **Data Files:** Data is stored in `csrankings-[a-z].csv` files, with authors alphabetized by first name.
*   **Contribution Guidelines:** Review `CONTRIBUTING.md` for details on how to contribute.
*   **Shallow Clone for Easy Contributions:** Use a shallow clone for quick edits without the full repository history.

## Running CS Rankings Locally

To set up and run the CS Rankings site locally, follow these steps:

1.  **Get the DBLP Data:** Run `make update-dblp` (requires ~19GiB of memory).
2.  **Build the Databases:** Run `make`.
3.  **Run a Local Web Server:** Use a command like `python3 -m http.server` and access the site at `http://0.0.0.0:8000`.
4.  **Install Dependencies:** Install the required dependencies:
    *   `libxml2-utils` (or equivalent)
    *   `npm`
    *   `typescript`
    *   `google-closure-compiler`
    *   `python-lxml`
    *   `pypy`
    *   `basex`

    You can install these using a command like:
    ```bash
    apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
    ```

## Acknowledgements

CS Rankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). It incorporates feedback from many contributors. The site is based on code and data collected by [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/) (UT-Austin) and the original faculty affiliation dataset constructed by [Papoutsaki et al.](http://cs.brown.edu/people/alexpap/faculty_dataset.html). The site uses information from [DBLP.org](http://dblp.org), made available under the ODC Attribution License.

## License

CS Rankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).