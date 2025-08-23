[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV.dev is a comprehensive, open-source vulnerability database designed to improve the security of open-source software, providing a centralized resource for vulnerability information.**

[View the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Centralized Vulnerability Database:**  A single source of truth for vulnerability information.
*   **API Access:**  Provides a robust API for accessing and querying vulnerability data.
*   **Scanner Tool:**  A Go-based tool to scan dependencies and check for known vulnerabilities via the OSV API (located in its own [repository](https://github.com/google/osv-scanner)).
*   **Web UI:** A user-friendly web interface is available at [https://osv.dev](https://osv.dev) for easy vulnerability exploration.
*   **Data Dumps:** Data dumps are available from a GCS bucket at `gs://osv-vulnerabilities`.
*   **Comprehensive Documentation:** Extensive documentation is available at [https://google.github.io/osv.dev](https://google.github.io/osv.dev) and [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)
*   **Community Support:** OSV has a mailing list ([https://groups.google.com/g/osv-discuss](https://groups.google.com/g/osv-discuss))

## Repository Structure

This repository contains the code for running the OSV platform on Google Cloud Platform (GCP). Key directories include:

*   `deployment/`: Deployment configurations using Terraform and Cloud Deploy.
*   `docker/`: Dockerfiles for various CI and deployment tasks.
*   `docs/`: Documentation files and build tools.
*   `gcp/api`: OSV API server files.
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Functions.
*   `gcp/indexer`: Version indexing components.
*   `gcp/website`: Backend and frontend for the OSV web interface.
*   `gcp/workers/`: Background workers for various tasks.
*   `osv/`: The core OSV Python library and supporting modules.
*   `tools/`: Utility scripts for development and maintenance.
*   `vulnfeeds/`:  Modules for vulnerability data conversion.

## Getting Started

To build locally, initialize submodules:

```bash
git submodule update --init --recursive
```

## Contributing

Contributions are welcome!  Learn how to contribute [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation).  For questions and suggestions, please [open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV integrates with many third-party tools.  These are community-built and not directly supported by OSV maintainers. Please consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software). Popular integrations include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)