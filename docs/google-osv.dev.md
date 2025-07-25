[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database and Infrastructure

**OSV (Open Source Vulnerability) provides a comprehensive database and infrastructure for open-source vulnerability information, helping developers identify and remediate security threats in their projects.**  Learn more and explore the code at the [OSV GitHub repository](https://github.com/google/osv.dev).

## Key Features

*   **Comprehensive Vulnerability Database:** Access a vast, curated database of known vulnerabilities affecting open-source projects.
*   **API Access:** Integrate OSV data into your tools and workflows with our robust API.
*   **Vulnerability Scanning Tool:** Identify vulnerabilities in your dependencies using our dedicated scanner ([osv-scanner](https://github.com/google/osv-scanner)).
*   **Web UI:** Explore and search the OSV database through our user-friendly web interface at <https://osv.dev>.
*   **Data Dumps:** Download data dumps for offline analysis and integration via a GCS bucket at `gs://osv-vulnerabilities`.

## Core Components & Architecture

This repository houses the infrastructure and codebase for the OSV platform, including:

*   **Deployment Configuration:** Terraform and Cloud Deploy configurations for deploying the OSV platform on Google Cloud Platform (GCP).
*   **API Server:** The OSV API server, providing access to vulnerability data.
*   **Web Interface Backend:** The backend for the osv.dev web interface, including blog functionality.
*   **Data Indexing:** Services for indexing and processing vulnerability data.
*   **Worker Processes:** Background workers for bisection, impact analysis, and other tasks.
*   **OSV Python Library:** The core Python library (`osv/`) used by many OSV services.
*   **Vulnerability Feeds:** Modules for converting vulnerability data from various sources.

## Getting Started

To build the OSV project locally, update and initialize all submodules.

```bash
git submodule update --init --recursive
```

For full documentation, please see <https://google.github.io/osv.dev>.

## Contributing

We welcome contributions!  Learn more about contributing code, data, and documentation in our [CONTRIBUTING.md](https://github.com/google/osv.dev/blob/main/CONTRIBUTING.md) file.

*   Join our [mailing list](https://groups.google.com/g/osv-discuss).
*   Report issues or suggest improvements by [opening an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV data is used by a vibrant community of developers.  Here are some notable third-party tools and integrations that utilize OSV:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)