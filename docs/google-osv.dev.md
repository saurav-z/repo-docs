[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database and Infrastructure

**OSV (Open Source Vulnerability) is a comprehensive database and infrastructure for tracking and sharing open-source vulnerabilities, helping developers identify and mitigate security risks.** Explore the official repository [here](https://github.com/google/osv.dev).

## Key Features

*   **Centralized Vulnerability Database:** A comprehensive, machine-readable database of open-source vulnerabilities.
*   **API Access:** Enables automated access and integration with security tools.
*   **Web UI:** Browse and search vulnerabilities at <https://osv.dev>.
*   **Data Dumps:** Access data dumps for offline analysis and integration.
*   **Dependency Scanning:** Integrates with tools like the [OSV Scanner](https://github.com/google/osv-scanner) to check dependencies for vulnerabilities.
*   **Supports Various Ecosystems:** Handles vulnerabilities in multiple ecosystems, including Go, Python, and more.

## Core Components

This repository contains the source code for the OSV infrastructure, including:

*   **Deployment:** Terraform & Cloud Deploy configuration files.
*   **Docker:** CI/CD and worker image definitions.
*   **Documentation:** Jekyll files for generating documentation.
*   **API:** OSV API server files (protobuf definitions, etc.).
*   **Datastore:** Configuration files for the Datastore.
*   **Workers:** Backend processes for vulnerability analysis, import, and data processing.
*   **Core Library:** The Python library (`osv/`) used across services.
*   **Vulnfeeds:** Modules for converting vulnerability feeds from various sources (NVD, Alpine, Debian).
*   **Website Backend:** The backend of the osv.dev web interface.

## Getting Started

To build locally, you'll need to initialize submodules:

```bash
git submodule update --init --recursive
```

## Resources

*   **Documentation:** [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
*   **API Documentation:** [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)
*   **Data Dumps:** Accessible from `gs://osv-vulnerabilities` (documentation: [https://google.github.io/osv.dev/data/#data-dumps](https://google.github.io/osv.dev/data/#data-dumps))
*   **OSV Scanner:** [https://github.com/google/osv-scanner](https://github.com/google/osv-scanner)
*   **Web UI:** <https://osv.dev>

## Contributing

Contributions are welcome! Learn more about contributing code, data, and documentation in [CONTRIBUTING.md](CONTRIBUTING.md).  Join the conversation on the [mailing list](https://groups.google.com/g/osv-discuss) or open an issue on [GitHub](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV integrates with several third-party tools. This list is not exhaustive and the listed tools are not endorsed or supported by the OSV maintainers. Please consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) for evaluation.  Some tools that integrate with OSV include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)