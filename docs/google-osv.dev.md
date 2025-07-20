[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV.dev: Open Source Vulnerability Database and Tools

**OSV.dev is an open-source project that provides a comprehensive vulnerability database and tools to help you identify and address security vulnerabilities in your software dependencies.**  You can find the original repository here: [https://github.com/google/osv.dev](https://github.com/google/osv.dev)

## Key Features

*   **Comprehensive Vulnerability Database:**  A centralized database of known vulnerabilities in open-source software.
*   **Dependency Scanning:**  Tools to scan your project's dependencies and check against the OSV database for known vulnerabilities.
*   **API Access:**  A robust API for accessing vulnerability data and integrating with other security tools.
*   **Web UI:** A user-friendly web interface for browsing vulnerabilities and related information.
*   **Data Dumps:** Regular data dumps available for offline analysis and integration.

## Core Components of the Repository

This repository contains the code necessary for running the OSV.dev project on Google Cloud Platform (GCP).  Key components include:

*   `deployment/`:  Configuration files for deployment using Terraform and Cloud Deploy.
*   `docker/`: Dockerfiles for CI/CD and worker image creation.
*   `docs/`: Documentation source files using Jekyll.
*   `gcp/api`: OSV API server code and protobuf definitions.
*   `gcp/datastore`: Datastore configuration.
*   `gcp/functions`: Cloud Functions for PyPI vulnerability publishing.
*   `gcp/indexer`: Version determination functionality.
*   `gcp/website`: Backend for the OSV.dev web interface.
*   `gcp/workers/`: Background worker processes for various tasks.
*   `osv/`: Core OSV Python library and related utilities.
*   `tools/`:  Development tools and scripts.
*   `vulnfeeds/`: Go modules for vulnerability data conversion.

**Note:** Many local building steps require submodules.  Ensure you run `git submodule update --init --recursive` after cloning.

## Documentation

*   **Comprehensive Documentation:**  [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
*   **API Documentation:**  [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)
*   **Data Dumps Information:**  [https://google.github.io/osv.dev/data/#data-dumps](https://google.github.io/osv.dev/data/#data-dumps)

## Using the Scanner

The OSV project provides a Go-based scanner to identify vulnerabilities in your dependencies.

*   **Scanning Capabilities:** Scans various lockfiles, Debian Docker containers, SBOMs (SPDX and CycloneDB), and Git repositories.
*   **Scanner Repository:** [https://github.com/google/osv-scanner](https://github.com/google/osv-scanner)

## Contributing

Contributions are welcome and encouraged!  Refer to the following resources:

*   **Contributing Code:** [CONTRIBUTING.md#contributing-code](https://github.com/google/osv.dev/blob/main/CONTRIBUTING.md#contributing-code)
*   **Contributing Data:** [CONTRIBUTING.md#contributing-data](https://github.com/google/osv.dev/blob/main/CONTRIBUTING.md#contributing-data)
*   **Contributing Documentation:** [CONTRIBUTING.md#contributing-documentation](https://github.com/google/osv.dev/blob/main/CONTRIBUTING.md#contributing-documentation)
*   **Mailing List:** [https://groups.google.com/g/osv-discuss](https://groups.google.com/g/osv-discuss)
*   **Open an Issue:** [https://github.com/google/osv.dev/issues](https://github.com/google/osv.dev/issues)

## Third-Party Tools and Integrations

A growing number of community-built tools leverage the OSV database.  These are not officially supported by the OSV maintainers. Review the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) to assess their suitability for your use.  Popular tools include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)