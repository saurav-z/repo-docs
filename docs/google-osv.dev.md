<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg">
</picture>

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database & Scanner

**OSV (Open Source Vulnerability) is a comprehensive, open-source vulnerability database and scanning tool designed to help you identify and address security vulnerabilities in your open-source dependencies.** Check out the original repo [here](https://github.com/google/osv.dev).

## Key Features

*   **Vulnerability Database:** A centralized, curated database of known vulnerabilities affecting open-source projects.
*   **Dependency Scanning:** A powerful scanner to identify vulnerable dependencies in your projects.
*   **Wide Ecosystem Support:** Scanning capabilities for various package managers and formats, including:
    *   Lockfiles
    *   Debian Docker containers
    *   SPDX and CycloneDB SBOMs
    *   Git repositories
*   **API Access:** Access the OSV database and scanner functionality through a well-defined API.
*   **Web UI:** User-friendly web interface for browsing vulnerabilities and project analysis: <https://osv.dev>.
*   **Data Dumps:** Access raw vulnerability data for offline analysis via Google Cloud Storage.

## Core Components of This Repository

This repository provides the foundational infrastructure and code for running the OSV platform, including:

*   **Deployment:** Terraform & Cloud Deploy configurations for infrastructure management.
*   **API Server:** Backend for the OSV API, including protobuf definitions and server logic.
*   **Web Interface:** Backend code for the OSV web interface.
*   **Data Processing:** Workers for data import, indexing, and analysis.
*   **Vulnerability Data:** Core Python library and data models.
*   **Scanner:** Integration with the OSV scanner for dependency analysis.
*   **Documentation:** Comprehensive documentation for the project.

### Directories

*   `deployment/`: Deployment configuration files
*   `docker/`: CI docker files
*   `docs/`: Documentation
*   `gcp/api`: OSV API server files
*   `gcp/datastore`: Datastore index file
*   `gcp/functions`: Cloud Function for PyPI vulnerabilities
*   `gcp/indexer`: The version determination indexer
*   `gcp/website`: OSV web interface backend
*   `gcp/workers/`: Workers for bisection and impact analysis
*   `osv/`: Core OSV Python library and models
*   `tools/`: Development scripts and tools
*   `vulnfeeds/`: NVD CVE conversion and feed converters

**Note:** To build locally, run `git submodule update --init --recursive`

## Contributing

Contributions are welcome! Learn more about contributing code, data, and documentation by reviewing the `CONTRIBUTING.md` file.

*   [Code Contributions](CONTRIBUTING.md#contributing-code)
*   [Data Contributions](CONTRIBUTING.md#contributing-data)
*   [Documentation Contributions](CONTRIBUTING.md#contributing-documentation)
*   [Mailing List](https://groups.google.com/g/osv-discuss)
*   [Open an Issue](https://github.com/google/osv.dev/issues)

## Third-Party Tools & Integrations

OSV integrates with a variety of third-party tools to enhance your vulnerability management workflow. The following is a list of popular tools that integrate with the OSV database and scanner. Note that these are community built tools and as such are not supported or endorsed by the core OSV maintainers.

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)