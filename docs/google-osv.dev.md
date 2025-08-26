<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg">
</picture>

---

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database and Infrastructure

**OSV is an open-source vulnerability database and infrastructure project designed to help you identify and address security vulnerabilities in your open-source dependencies.**  This repository contains the core infrastructure for the OSV project, providing the foundation for a robust and reliable vulnerability management system. [Learn more about OSV on GitHub](https://github.com/google/osv.dev).

## Key Features

*   **Comprehensive Vulnerability Database:**  Access a curated database of known vulnerabilities in open-source software.
*   **Dependency Scanning:**  Utilize the OSV scanner to identify vulnerable dependencies in your projects.
*   **API Access:** Integrate the OSV API into your security workflows for automated vulnerability checks.
*   **Data Dumps:**  Download data dumps for offline analysis and integration with your own tools.
*   **Web UI:**  Browse and search the OSV database through a user-friendly web interface.
*   **Community Driven:** OSV thrives on community contributions, encouraging collaboration and knowledge sharing.

## Key Components of this Repository

This repository contains the code and configuration for the OSV infrastructure, including:

*   **Deployment:** Terraform and Cloud Deploy configuration for deployment on GCP.
*   **API Server:** The OSV API server, serving vulnerability data.
*   **Web UI Backend:** The backend code for the OSV web interface.
*   **Data Processing Workers:**  Workers for tasks such as bisection, import, and export.
*   **Indexer:**  The component responsible for determining the version
*   **OSV Python Library:** The core OSV Python library, used in many Python services.
*   **Vulnerability Feed Converters:** Modules for converting vulnerability data from various sources.

*   **Docs:**  Jekyll files for https://google.github.io/osv.dev/ and API documentation.
*   **Tools:**  Misc scripts/tools, mostly intended for development (datastore stuff, linting)

## Getting Started

To build locally, you need to check out submodules as well:

```bash
git submodule update --init --recursive
```

## Documentation

Comprehensive documentation is available at:
*   [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
*   API Documentation: [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)

## Data Dumps

Data dumps are available from a GCS bucket: `gs://osv-vulnerabilities`.  More information: [https://google.github.io/osv.dev/data/#data-dumps](https://google.github.io/osv.dev/data/#data-dumps).

## Web UI

Explore the OSV web UI at: [https://osv.dev](https://osv.dev).

## OSV Scanner

The OSV scanner is a powerful Go-based tool that scans your dependencies against the OSV database. It supports various formats, including lockfiles, Debian Docker containers, SBOMs (SPDX and CycloneDX), and Git repositories. The scanner is located in its [own repository](https://github.com/google/osv-scanner).

## Contributing

Contributions are welcome!  Learn more about contributing to [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation). You can also join the [mailing list](https://groups.google.com/g/osv-discuss).

## Third-Party Tools and Integrations

OSV integrates with a variety of third-party tools. Check out:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

Please note that these tools are community-built and are not supported or endorsed by the OSV maintainers.