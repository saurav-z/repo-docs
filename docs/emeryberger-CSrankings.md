# CSrankings: Your Definitive Guide to Top Computer Science Schools

**CSrankings provides a data-driven, metrics-based ranking of top Computer Science schools based on faculty publications in highly selective conferences.**

[View the original repository on GitHub](https://github.com/emeryberger/CSrankings)

## Key Features:

*   **Metrics-Based Ranking:** Ranks institutions based on the number of publications by faculty in top Computer Science conferences.
*   **Data-Driven Approach:** Leverages publication data, offering a more objective evaluation compared to survey-based rankings.
*   **Difficult-to-Game Methodology:** Focuses on publications in selective conferences, making it resistant to manipulation.
*   **Comprehensive Data:** Includes an extensive dataset, drawing from DBLP for publications and faculty affiliations.
*   **Regular Updates:** The ranking is updated quarterly to reflect the latest research output.
*   **Community-Driven:**  Contributors can submit pull requests to update faculty information.

## How CSrankings Works

Unlike rankings that rely on surveys, CSrankings employs a metrics-based approach. This method focuses on measuring the research output of faculty members at different institutions. The ranking algorithm primarily considers the number of publications in leading computer science conferences.

## Contributing to CSrankings

Contributions are welcome!

*   **Updating Affiliations:** Contribute by editing the `csrankings-[a-z].csv` files.
*   **Contribution Guidelines:** Review the `CONTRIBUTING.md` file for detailed instructions.
*   **Submission Process:** Submit pull requests, understanding that updates are processed quarterly.
*   **Shallow Clone for Quick Contributions:** Utilize a shallow clone for faster and easier contribution to the project.

## Setup and Local Usage

To run the website locally:

1.  Download the DBLP data: `make update-dblp` (requires ~19GB of memory).
2.  Rebuild databases: `make`.
3.  Run a local web server (e.g., `python3 -m http.server`).
4.  Access the site at [http://0.0.0.0:8000](http://0.0.0.0:8000).

**Dependencies:** Install necessary packages using:
```bash
apt-get install libxml2-utils npm python-lxml basex; npm install -g typescript google-closure-compiler
```

## Acknowledgements

CSrankings was primarily developed and is maintained by [Emery Berger](https://emeryberger.com). The project draws on data and insights from various sources:

*   **Initial Data:** Based on code and data collected by Swarat Chaudhuri (UT-Austin).
*   **Faculty Dataset:**  Originally constructed by Papoutsaki et al.
*   **DBLP:** Utilizes information from [DBLP.org](http://dblp.org), licensed under the ODC Attribution License.

## License

CSrankings is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).