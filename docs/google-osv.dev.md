[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database & Infrastructure

**OSV (Open Source Vulnerability) provides a comprehensive, centralized database and infrastructure for open-source vulnerability information.** This repository houses the core components that power the OSV platform. You can explore the live web UI at <https://osv.dev>.

[Visit the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Centralized Vulnerability Database:** Access a vast collection of open-source vulnerability data.
*   **API Access:** Integrate with the OSV API to retrieve vulnerability information programmatically.
*   **Web UI:** Browse and search the OSV database through an intuitive web interface.
*   **Dependency Scanning:** Use the OSV scanner (available in its [own repository](https://github.com/google/osv-scanner)) to check your project's dependencies for known vulnerabilities.
*   **Data Dumps:** Download comprehensive data dumps for offline analysis and integration (available at `gs://osv-vulnerabilities`).

## Repository Structure

This repository contains the code and configuration for the OSV platform, including:

*   `deployment/`: Infrastructure-as-code for deployment.
*   `docker/`: Dockerfiles for CI and deployment.
*   `docs/`: Documentation files and API documentation generation tools.
*   `gcp/api`: OSV API server files.
*   `gcp/datastore`: Datastore configuration.
*   `gcp/functions`: Cloud Functions for vulnerability data ingestion.
*   `gcp/indexer`: Version indexing components.
*   `gcp/website`: Backend for the OSV web interface.
*   `gcp/workers/`: Background workers for various tasks (bisection, import, export).
*   `osv/`: The core OSV Python library.
*   `tools/`: Development tools and scripts.
*   `vulnfeeds/`: Tools for converting vulnerability data from various sources.

**Important:** To build locally, run `git submodule update --init --recursive` after cloning the repository.

## Documentation

*   **Comprehensive Documentation:** <https://google.github.io/osv.dev>
*   **API Documentation:** <https://google.github.io/osv.dev/api/>
*   **Data Dumps Documentation:** <https://google.github.io/osv.dev/data/#data-dumps>

## Contributing

We welcome contributions!  Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing code, data, or documentation.

*   **Mailing List:** [OSV Discuss](https://groups.google.com/g/osv-discuss)
*   **Open an Issue:** [GitHub Issues](https://github.com/google/osv.dev/issues)

## Third-Party Tools & Integrations

OSV integrates with various third-party tools. These are community-built and not officially supported or endorsed by OSV maintainers.

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)