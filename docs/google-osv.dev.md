<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg">
</picture>

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV (Open Source Vulnerability) is a comprehensive database and API for open-source vulnerabilities, empowering developers to identify and address security threats quickly.**  This repository houses the infrastructure powering the OSV project.  Explore the official OSV documentation and API at [https://google.github.io/osv.dev](https://google.github.io/osv.dev).

**Key Features:**

*   **Comprehensive Vulnerability Data:** Access a centralized and structured database of open-source vulnerabilities.
*   **API Access:**  Programmatically query and integrate vulnerability data into your security workflows.
*   **Vulnerability Scanner:** Identify vulnerabilities in your projects using the [OSV Scanner](https://github.com/google/osv-scanner).
*   **Data Dumps:** Download data dumps for offline analysis and integration.
*   **Web UI:** Easily browse and explore vulnerabilities through the OSV web interface at <https://osv.dev>.
*   **Community-Driven:** Benefit from an active community and contribute to the project.

## Project Structure

This repository contains the code for running the OSV infrastructure on Google Cloud Platform (GCP), including:

*   **`deployment/`:** Infrastructure-as-code configuration using Terraform and Cloud Deploy.
*   **`docker/`:** Dockerfiles for CI, deployment, and worker images.
*   **`docs/`:** Documentation files for the OSV website.
*   **`gcp/api`:**  OSV API server files, including Protobuf definitions.
*   **`gcp/datastore`:** Datastore index configuration.
*   **`gcp/functions`:** Cloud Functions for processing vulnerability data (e.g., PyPI).
*   **`gcp/indexer`:** Vulnerability indexing components.
*   **`gcp/website`:** Backend for the OSV web interface.
*   **`gcp/workers/`:** Background workers for tasks like bisection and impact analysis.
*   **`osv/`:** Core OSV Python library and related utilities.
*   **`tools/`:** Development scripts and utilities.
*   **`vulnfeeds/`:** Modules for converting vulnerability data from various sources.

To build locally, be sure to initialize submodules:

```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions! Learn how to contribute to the [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation).

Have questions or suggestions? Please [open an issue](https://github.com/google/osv.dev/issues) or join the [mailing list](https://groups.google.com/g/osv-discuss).

## Third-Party Tools and Integrations

OSV is supported by many third-party tools.  *These tools are community-maintained and are not officially supported or endorsed by the OSV maintainers*.  Consider the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) when selecting tools for your needs. Examples include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

[**Visit the OSV GitHub Repository**](https://github.com/google/osv.dev)