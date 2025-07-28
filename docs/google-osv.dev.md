[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database & Infrastructure

**OSV (Open Source Vulnerability) provides a comprehensive, centralized database of open-source vulnerabilities, along with tools and infrastructure to help you manage and mitigate risks.**  Learn more and contribute to the project on [GitHub](https://github.com/google/osv.dev).

## Key Features

*   **Centralized Vulnerability Database:** A single source of truth for known vulnerabilities in open-source projects.
*   **API Access:** Easily integrate OSV data into your security workflows and tools.  See the [API documentation](https://google.github.io/osv.dev/api/).
*   **Vulnerability Scanning Tool:**  A Go-based scanner to check your project's dependencies against the OSV database.  See the scanner repo [here](https://github.com/google/osv-scanner).
*   **Web UI:** Explore vulnerabilities through a user-friendly web interface available at <https://osv.dev>.
*   **Data Dumps:** Access raw vulnerability data for offline analysis and integration.  Data dumps are available in a GCS bucket: `gs://osv-vulnerabilities`.  See the [documentation](https://google.github.io/osv.dev/data/#data-dumps) for more details.

## Project Structure Overview

This repository contains the infrastructure and code that powers the OSV platform, including:

*   **Deployment Configuration:** Terraform & Cloud Deploy configurations for deployment.
*   **Docker Images:**  Dockerfiles for CI and deployment processes, and base worker images.
*   **Documentation:** Jekyll files for the OSV documentation site ([https://google.github.io/osv.dev/](https://google.github.io/osv.dev/))
*   **API Server:**  Code for the OSV API server (gcp/api) including Protobuf files.
*   **Data Processing & Indexing:** Components for data storage, indexing, and processing, including Cloud Functions and Workers.
*   **Web Interface Backend:**  The backend code for the OSV web interface (gcp/website).
*   **Core OSV Library:**  The core Python library for interacting with the OSV ecosystem (osv/).
*   **Vulnerability Feed Converters:**  Tools for converting vulnerability data from various sources (vulnfeeds/).

To build locally, you may need to run:

```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions!  Learn more about contributing [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation).

*   **Mailing List:** Join the [mailing list](https://groups.google.com/g/osv-discuss).
*   **Report Issues:**  [Open an issue](https://github.com/google/osv.dev/issues) to report a problem or suggest an improvement.

## Third-Party Tools & Integrations

OSV integrates with a variety of third-party security tools. These tools are community-maintained and are not officially supported or endorsed by the OSV project.  Please review the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) before using.

Examples include:

*   Cortex XSOAR
*   dep-scan
*   Dependency-Track
*   GUAC
*   OSS Review Toolkit
*   pip-audit
*   Renovate
*   Trivy