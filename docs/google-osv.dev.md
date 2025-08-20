[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: The Open Source Vulnerability Database

**OSV is a free, open-source vulnerability database and API that helps you identify and remediate security vulnerabilities in open-source software.**

[Go to the original repository](https://github.com/google/osv.dev)

## Key Features

*   **Comprehensive Vulnerability Data:** OSV provides a centralized database of known vulnerabilities affecting open-source projects.
*   **API Access:** Access the OSV database programmatically using a robust API for easy integration into your security tools.
*   **Dependency Scanning:** The OSV scanner helps you identify vulnerable dependencies in your projects.
*   **Web UI:** Browse and explore vulnerabilities through a user-friendly web interface.
*   **Data Dumps:** Download data dumps for offline analysis and integration.

## Documentation

*   Comprehensive documentation is available at: [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
*   API documentation is available at: [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)
*   Data Dumps: [https://google.github.io/osv.dev/data/#data-dumps](https://google.github.io/osv.dev/data/#data-dumps)

## Using the Scanner

The OSV scanner, a Go-based tool, checks your project's dependencies against the OSV database. It supports:

*   Various lockfiles
*   Debian Docker containers
*   SPDX and CycloneDB SBOMs
*   Git repositories

The scanner is located in its [own repository](https://github.com/google/osv-scanner).

## Repository Structure

This repository contains the code for running the OSV platform on Google Cloud Platform (GCP). Key directories include:

*   `deployment/`: Terraform and Cloud Deploy configuration.
*   `docker/`: Dockerfiles for CI and deployment.
*   `docs/`: Jekyll files for the OSV documentation site.
*   `gcp/api`: OSV API server files.
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Function for publishing PyPI vulnerabilities.
*   `gcp/indexer`: Version determination indexer.
*   `gcp/website`: Backend of the OSV web interface.
*   `gcp/workers/`: Background workers for various tasks (bisection, import, export, etc.).
*   `osv/`: Core OSV Python library and ecosystem helpers.
*   `tools/`: Development and utility scripts.
*   `vulnfeeds/`: Go module for NVD CVE conversion and feed converters.

## Getting Started

Ensure you initialize the submodules for local builds:

```bash
git submodule update --init --recursive
```

## Contributing

Contributions are welcome! Learn more about contributing code, data, and documentation via the following links:

*   [Code Contributions](CONTRIBUTING.md#contributing-code)
*   [Data Contributions](CONTRIBUTING.md#contributing-data)
*   [Documentation Contributions](CONTRIBUTING.md#contributing-documentation)

For questions and suggestions, please [open an issue](https://github.com/google/osv.dev/issues) or join the [mailing list](https://groups.google.com/g/osv-discuss).

## Third-Party Tools and Integrations

The OSV database is used by several community tools. Note that these are community-built tools and are not supported or endorsed by the core OSV maintainers.

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)