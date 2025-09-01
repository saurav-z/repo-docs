<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg">
</picture>

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: The Open Source Vulnerability Database and Scanner

**OSV (Open Source Vulnerability) is a comprehensive database and a suite of tools designed to help you identify and mitigate vulnerabilities in your open-source dependencies.** This repository houses the infrastructure powering the OSV platform, providing vital resources for software security.

[Visit the official OSV repository on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Comprehensive Vulnerability Database:** Access a curated, up-to-date database of known vulnerabilities affecting open-source software.
*   **Dependency Scanning:** Utilize the OSV scanner to identify vulnerable dependencies in your projects.
*   **API Access:** Integrate OSV data and functionality into your own tools and workflows through the OSV API.
*   **Web UI:** Explore vulnerabilities and project information through the user-friendly OSV web interface.
*   **Data Dumps:** Access data dumps for offline analysis and integration into security tools.

## Core Components of the OSV Platform

This repository contains the essential components that power the OSV platform, including:

*   **API Server:** The OSV API server, providing access to vulnerability data.
*   **Web UI Backend:** The backend infrastructure for the OSV website.
*   **Data Processing Workers:** Workers for tasks like bisection and impact analysis.
*   **Indexer:** The component responsible for determining the versions affected by vulnerabilities.
*   **Documentation:** Comprehensive documentation to help you get started.
*   **Vulnerability Feed Converters:** Tools for converting vulnerability data from various sources (e.g., NVD, Debian, Alpine).

## Key Directories

| Directory       | Description                                                                     |
|-----------------|---------------------------------------------------------------------------------|
| `deployment/`   | Terraform & Cloud Deploy configuration files.                                  |
| `docker/`       | Docker files for CI, deployment, and worker images.                             |
| `docs/`         | Jekyll files for the OSV documentation site.                                    |
| `gcp/api`       | OSV API server files and Protobuf definitions.                                  |
| `gcp/datastore` | Datastore configuration files.                                                |
| `gcp/functions` | Cloud Functions for PyPI vulnerability publishing.                                |
| `gcp/indexer`   | The version determination indexer.                                                 |
| `gcp/website`   | Backend of the OSV web interface.                                          |
| `gcp/workers/`  | Workers for bisection, impact analysis, and data processing.                       |
| `osv/`          | The core OSV Python library.                                                    |
| `tools/`        | Development scripts and tools.                                                 |
| `vulnfeeds/`    | Go module for NVD CVE conversion and other feed converters.                      |

## Getting Started

To build locally, make sure you initialize all submodules using:

```bash
git submodule update --init --recursive
```

## Contribute

We welcome contributions! Learn more about contributing code, data, and documentation through the links below:
*   [Contributing Code](CONTRIBUTING.md#contributing-code)
*   [Contributing Data](CONTRIBUTING.md#contributing-data)
*   [Contributing Documentation](CONTRIBUTING.md#contributing-documentation)

Join the conversation on our [mailing list](https://groups.google.com/g/osv-discuss) or [open an issue](https://github.com/google/osv.dev/issues) if you have a question or suggestion.

## Third-Party Tools and Integrations

A wide range of community-built tools and integrations leverage the OSV database. Please note that these tools are not supported or endorsed by the core OSV maintainers. For guidance, please consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software). Some popular third-party tools include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)