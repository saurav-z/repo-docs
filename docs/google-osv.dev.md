<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg">
</picture>

---

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV (Open Source Vulnerability) is a comprehensive, open-source database and API designed to track and provide information about vulnerabilities in open-source software.** This repository ([https://github.com/google/osv.dev](https://github.com/google/osv.dev)) contains the code that powers the OSV platform.

## Key Features

*   **Centralized Vulnerability Database:** A single source of truth for open-source vulnerability information.
*   **API Access:** Allows programmatic access to vulnerability data for integration with other tools.
*   **Vulnerability Scanning Tools:**  Integrates with the [OSV Scanner](https://github.com/google/osv-scanner) to scan dependencies and identify vulnerabilities.
*   **Web UI:**  A user-friendly web interface ([https://osv.dev](https://osv.dev)) to browse and search for vulnerabilities.
*   **Data Dumps:** Provides data dumps for offline analysis and integration into various systems, available from a GCS bucket.

## Core Components of the OSV Platform

This repository houses the essential components that comprise the OSV platform, including:

*   **API Server:** The OSV API server, handling requests and serving vulnerability data.
*   **Web Interface Backend:** The backend code for the OSV web interface.
*   **Data Indexing:** Processes vulnerability information and prepares it for efficient searching and retrieval.
*   **Data Processing Workers:** Background workers for bisection, impact analysis, importing data, and database maintenance.
*   **Core OSV Library:**  A Python library providing essential functions for vulnerability handling and data manipulation.
*   **Vulnerability Feed Conversion:** Modules for converting vulnerability feeds from various sources (e.g., NVD, Debian).

## Getting Started

To get started, you'll need to check out the submodules:

```bash
git submodule update --init --recursive
```

Detailed documentation is available at [https://google.github.io/osv.dev](https://google.github.io/osv.dev). API documentation can be found at [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/).

## Contributing

We welcome contributions from the community!  Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on contributing code, data, and documentation.  You can also join the [mailing list](https://groups.google.com/g/osv-discuss) and [open an issue](https://github.com/google/osv.dev/issues) with any questions or suggestions.

## Third-Party Tools and Integrations

OSV is supported by a number of third-party tools and integrations.  Note these are community-built and not officially endorsed by the OSV project:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)