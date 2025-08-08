[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: The Open Source Vulnerability Database

**OSV (Open Source Vulnerability) is a free and open database that provides a comprehensive and standardized way to track and manage vulnerabilities in open-source software.** This repository houses the infrastructure and code that powers the OSV platform. Find out more at the [original repository](https://github.com/google/osv.dev).

## Key Features:

*   **Comprehensive Vulnerability Data:** Access a centralized database of known vulnerabilities affecting open-source software.
*   **Standardized Format:** Utilize a consistent and structured data format for easy analysis and integration.
*   **API Access:** Integrate OSV data into your security tools and workflows with our robust API.
*   **Web UI:** Easily browse and search for vulnerabilities through the OSV web interface at <https://osv.dev>.
*   **Dependency Scanning Tool:** Scan your project dependencies and check against the OSV database for vulnerabilities using our Go-based scanner.
*   **Data Dumps:** Access data dumps from a GCS bucket for offline analysis and integration.

## Core Components:

This repository contains the following key components that make up the OSV platform:

*   **Deployment Infrastructure:** Terraform & Cloud Deploy config files for deploying the OSV platform on GCP.
*   **API Server:** The OSV API server files, including protocol buffer definitions.
*   **Web UI Backend:** The backend code for the OSV web interface.
*   **Data Processing Workers:** Workers for bisection, impact analysis, importing data, exporting data, and alias management.
*   **Core Python Library:** The `osv` Python library used across various services.
*   **Indexer:** The determine version indexer
*   **Vulnerability Feed Converters:** Tools for converting vulnerability feeds from various sources (e.g., NVD, Alpine, Debian).
*   **Documentation:** Jekyll files for generating the OSV documentation website, as well as API documentation.

For a detailed breakdown of the directory structure, refer to the original README.

## Getting Started

To get started with OSV, explore our comprehensive documentation.

*   **Documentation:** [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
*   **API Documentation:** [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)

## Using the Scanner

The OSV scanner is a Go-based tool that scans your dependencies and checks them against the OSV database for known vulnerabilities.  Learn more in [the scanner's repository](https://github.com/google/osv-scanner).

## Contributing

We welcome contributions! See our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on contributing code, data, and documentation.

*   **Mailing List:** [OSV Discussion](https://groups.google.com/g/osv-discuss)
*   **Issue Tracker:** [Open an Issue](https://github.com/google/osv.dev/issues)

## Third-Party Tools and Integrations

OSV integrates with several third-party tools. Note that these are community-built tools and are not supported or endorsed by the core OSV maintainers.

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)