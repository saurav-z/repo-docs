# OSV: Open Source Vulnerability Database

**OSV (Open Source Vulnerability) is a free, open-source database and API designed to provide comprehensive vulnerability information for open-source software, empowering developers to identify and mitigate security risks.**  Explore the OSV project on [GitHub](https://github.com/google/osv.dev).

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

## Key Features:

*   **Centralized Vulnerability Database:** Access a unified, comprehensive database of known vulnerabilities in open-source projects.
*   **Open API:** Integrate OSV data seamlessly into your security tools and workflows through a well-documented API.
*   **Dependency Scanning Tools:** Scan your project dependencies to identify vulnerable components using the OSV scanner, which supports various package managers and formats.
*   **Data Dumps:** Download vulnerability data dumps for offline analysis and integration.
*   **Web UI:** Explore the OSV database and browse vulnerabilities through a user-friendly web interface:  <https://osv.dev>

## Key Resources

*   **Documentation:**  Comprehensive documentation can be found [here](https://google.github.io/osv.dev).
*   **API Documentation:** Access detailed API documentation [here](https://google.github.io/osv.dev/api/).
*   **Data Dumps:**  Download data dumps from a GCS bucket at `gs://osv-vulnerabilities` ([documentation](https://google.github.io/osv.dev/data/#data-dumps)).
*   **OSV Scanner:**  Scan your project dependencies using the [OSV scanner](https://github.com/google/osv-scanner).

## Repository Structure:

This repository contains the code for running the OSV platform on Google Cloud Platform (GCP), encompassing:

*   **Deployment Configuration:** Terraform and Cloud Deploy configurations.
*   **CI/CD:** Docker files for CI/CD pipelines.
*   **Documentation:** Jekyll files for the OSV documentation site.
*   **API Server:**  OSV API server files (Go & Protobuf definitions).
*   **Database Indexing & Workers:** Components for data indexing, importing, and analysis.
*   **Web Interface Backend:** Backend code for the OSV web interface.
*   **Core OSV Library:** The core Python library used throughout the project.
*   **Vulnerability Feed Conversion:** Modules for converting vulnerability feeds (e.g., NVD, Alpine, Debian).

To build locally, initialize submodules:

```bash
git submodule update --init --recursive
```

## Contributing

We encourage contributions! Learn how to contribute [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation).

*   **Mailing List:** [OSV Discuss](https://groups.google.com/g/osv-discuss)
*   **Issue Tracker:** [Open an Issue](https://github.com/google/osv.dev/issues)

## Third-Party Tools and Integrations

OSV integrates with various community-developed tools.  Please consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) before using these tools.  Examples include:

*   Cortex XSOAR
*   dep-scan
*   Dependency-Track
*   GUAC
*   OSS Review Toolkit
*   pip-audit
*   Renovate
*   Trivy