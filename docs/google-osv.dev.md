[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV provides a comprehensive, open-source database of vulnerabilities affecting open-source software, empowering developers to identify and address security risks effectively.**

[Visit the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Comprehensive Vulnerability Data:**  A curated database of known vulnerabilities across various open-source projects.
*   **OSV API:**  A powerful API to query the vulnerability database and integrate with your security tools.
*   **Web UI:**  A user-friendly web interface to browse and explore vulnerabilities: <https://osv.dev>
*   **Scanning Tool:** A Go-based scanner tool to check dependencies against the OSV database.
*   **Data Dumps:** Access data dumps for offline analysis and integration with other security solutions.

## Core Components

This repository houses the core components of the OSV infrastructure:

*   **`deployment/`:** Terraform and Cloud Deploy configuration files.
*   **`docker/`:** Dockerfiles for CI and worker images.
*   **`docs/`:** Jekyll files for the OSV documentation.
*   **`gcp/api`:** OSV API server files.
*   **`gcp/datastore`:** Datastore index configuration.
*   **`gcp/functions`:** Cloud Functions for PyPI vulnerability publishing.
*   **`gcp/indexer`:** The version determination indexer.
*   **`gcp/website`:** Backend for the OSV web interface, including blog content.
*   **`gcp/workers/`:** Background workers for data processing (bisection, import, export, alias).
*   **`osv/`:** The core OSV Python library and ecosystem helpers.
*   **`tools/`:** Development scripts and tools.
*   **`vulnfeeds/`:** Modules for converting vulnerability feeds (e.g., NVD, Alpine, Debian).

## Getting Started

To build locally, you'll need to initialize the submodules:

```bash
git submodule update --init --recursive
```

## Contributing

Contributions are welcome!  Please see the following for contribution guidelines:

*   [Contributing Code](CONTRIBUTING.md#contributing-code)
*   [Contributing Data](CONTRIBUTING.md#contributing-data)
*   [Contributing Documentation](CONTRIBUTING.md#contributing-documentation)

Have questions or suggestions?  [Open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV integrates with a variety of third-party tools and services. *Note: These are community-built tools and are not officially supported or endorsed by the OSV maintainers.*

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

## Documentation

*   Comprehensive documentation is available [here](https://google.github.io/osv.dev).
*   API documentation is available [here](https://google.github.io/osv.dev/api/).
*   Learn more about Data Dumps [here](https://google.github.io/osv.dev/data/#data-dumps).