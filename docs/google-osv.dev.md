[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV provides a comprehensive and centralized database of open-source vulnerabilities, empowering developers to proactively identify and address security risks in their projects.**  This repository contains the infrastructure for running the OSV platform.

[View the OSV project on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Centralized Vulnerability Database:** A comprehensive and curated database of open-source vulnerabilities.
*   **Dependency Scanning:**  Tools available for scanning your dependencies and checking for vulnerabilities.  See the [OSV scanner](https://github.com/google/osv-scanner) for details.
*   **Web UI:**  An accessible web interface to browse and search the OSV database at <https://osv.dev>.
*   **API Access:**  Programmatic access to the OSV data via a robust API.
*   **Data Dumps:** Downloadable data dumps for offline analysis and integration.
*   **Community Contributions:**  Open to contributions for data, code, and documentation.

## Core Components of this Repository

This repository houses the core infrastructure for the OSV platform, including:

*   **Deployment Configuration:** Terraform and Cloud Deploy configurations for deployment on GCP.
*   **API Server:**  The OSV API server implementation.
*   **Web Interface Backend:**  Backend code for the OSV web interface.
*   **Data Processing Workers:**  Workers for tasks like vulnerability indexing and impact analysis.
*   **Vulnerability Feed Converters:** Tools to convert vulnerability data from various sources (e.g., NVD, Alpine, Debian).
*   **Core Python Library:**  The core OSV Python library used across the platform.

**Key Directories:**
*   `deployment/`: Deployment configuration
*   `docker/`:  CI/CD docker files
*   `docs/`: Documentation
*   `gcp/api`: OSV API server files
*   `gcp/datastore`: Datastore index files
*   `gcp/functions`: PyPI vulnerability publishing Cloud Function
*   `gcp/indexer`: Version indexing tools
*   `gcp/website`:  Backend of the web interface
*   `gcp/workers/`: Workers for bisection and impact analysis
*   `osv/`:  The core OSV Python library
*   `tools/`:  Development scripts and tools
*   `vulnfeeds/`:  Vulnerability feed converters

## Getting Started

To set up this project locally, you'll need to initialize submodules:

```bash
git submodule update --init --recursive
```

## Contributing

We encourage contributions!  Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on contributing code, data, and documentation.  Join the community on the [mailing list](https://groups.google.com/g/osv-discuss) or open an issue to ask a question or make a suggestion.

## Third-Party Tools & Integrations

The OSV database is used by a growing ecosystem of third-party tools.  Note that these are community-built and not officially supported by the OSV project.

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)