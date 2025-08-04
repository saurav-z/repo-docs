[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV.dev: Open Source Vulnerability Database

**OSV.dev is a comprehensive, open-source database of vulnerabilities in open-source software, designed to help developers and security professionals identify and mitigate risks.**  ([Visit the original repository](https://github.com/google/osv.dev))

## Key Features

*   **Centralized Vulnerability Database:**  A single source of truth for vulnerabilities across various open-source ecosystems.
*   **Web UI:** User-friendly interface for searching and exploring vulnerability data: <https://osv.dev>
*   **API Access:**  Programmatic access to vulnerability data for integration with security tools.
*   **Data Dumps:** Availability of data dumps for offline analysis and integration:  `gs://osv-vulnerabilities` (See [documentation](https://google.github.io/osv.dev/data/#data-dumps))
*   **Dependency Scanning Tool:** A Go-based scanner to identify vulnerabilities in your dependencies ([osv-scanner](https://github.com/google/osv-scanner)).
*   **Go-based scanner:**  Scans various lockfiles, Debian Docker containers, SBOMs, and Git repositories.

## Project Structure

This repository contains the code that powers [osv.dev](https://osv.dev), including:

*   `deployment/`:  Terraform and Cloud Deploy configurations.
*   `docker/`:  Docker files for CI and deployment.
*   `docs/`:  Documentation files built with Jekyll.
*   `gcp/api`:  OSV API server files.
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Functions for PyPI vulnerability publishing.
*   `gcp/indexer`: Version determination indexer.
*   `gcp/website`:  Backend for the osv.dev web interface.
*   `gcp/workers/`:  Background workers for bisection, import, and export.
*   `osv/`:  The core OSV Python library.
*   `tools/`:  Development scripts and tools.
*   `vulnfeeds/`:  Go module for vulnerability feed conversions (e.g., NVD, Alpine, Debian).

**Important:**  Remember to initialize submodules for local building: `git submodule update --init --recursive`

## Contributing

We welcome contributions! Learn more about contributing [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation).
Join the discussion on our [mailing list](https://groups.google.com/g/osv-discuss).  Have a question or suggestion?  [Open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

A vibrant community has built tools and integrations using OSV.  Note that these are community-built and not officially supported by the OSV maintainers.  Consider consulting the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) to determine their suitability.  Examples include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)