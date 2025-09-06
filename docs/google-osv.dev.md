<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg" alt="OSV Logo">
</picture>

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database & Tools

**OSV (Open Source Vulnerability) is a free, open-source database and toolkit designed to improve the security of open-source software by providing a centralized, reliable source of vulnerability information.**  Visit the [OSV GitHub repository](https://github.com/google/osv.dev) to learn more.

## Key Features of OSV

*   **Comprehensive Vulnerability Database:**  A centralized database of known vulnerabilities, providing a single source of truth for security information.
*   **Dependency Scanning:** Tools for scanning your project dependencies against the OSV database to identify vulnerabilities.
*   **Wide Ecosystem Support:**  Supports scanning various lockfiles, SBOMs (SPDX, CycloneDX), Debian Docker containers, and Git repositories.
*   **API Access:** Provides an API for programmatic access to vulnerability data, allowing for seamless integration into existing security workflows.
*   **Web UI:**  A user-friendly web interface (<https://osv.dev>) for browsing and searching the vulnerability database.
*   **Community-Driven:**  Encourages contributions of code, data, and documentation to improve the database and tools.

## Core Components of the OSV Project

This repository contains the core components that power the OSV infrastructure on Google Cloud Platform (GCP):

*   **API Server:** Serves the OSV API.
*   **Web Interface Backend:** Backend code for the OSV web interface.
*   **Data Indexing & Processing:**  Components for indexing and processing vulnerability data.
*   **Vulnerability Data Feeds:** Modules for converting and integrating vulnerability data from various sources (e.g., NVD, Debian, Alpine).
*   **Workers:** Background processes for tasks such as bisection and impact analysis.

**Key Directories:**

*   `deployment/`: Infrastructure deployment configuration (Terraform, Cloud Build).
*   `docker/`: Docker files for CI and other processes.
*   `docs/`: Documentation files (Jekyll, API docs).
*   `gcp/`: API server, datastore, functions, indexer, web interface backend, and workers.
*   `osv/`: Core OSV Python library and datastore model definitions.
*   `tools/`: Development scripts and tools.
*   `vulnfeeds/`: Modules for converting vulnerability feeds.

**To build locally, you'll need to check out submodules:**

```bash
git submodule update --init --recursive
```

## Documentation and Resources

*   **Comprehensive Documentation:** [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
*   **API Documentation:** [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)
*   **Data Dumps:**  Available in a GCS bucket: `gs://osv-vulnerabilities`.  See documentation for details:  [https://google.github.io/osv.dev/data/#data-dumps](https://google.github.io/osv.dev/data/#data-dumps)
*   **Web UI:** <https://osv.dev>
*   **Scanner:** [https://github.com/google/osv-scanner](https://github.com/google/osv-scanner)

## Contributing

We welcome contributions to OSV!  Learn how to contribute:

*   **Code:** [CONTRIBUTING.md#contributing-code](CONTRIBUTING.md#contributing-code)
*   **Data:** [CONTRIBUTING.md#contributing-data](CONTRIBUTING.md#contributing-data)
*   **Documentation:** [CONTRIBUTING.md#contributing-documentation](CONTRIBUTING.md#contributing-documentation)
*   **Mailing List:** [https://groups.google.com/g/osv-discuss](https://groups.google.com/g/osv-discuss)
*   **Issues & Suggestions:**  [Open an issue](https://github.com/google/osv.dev/issues)

## Third-Party Tools & Integrations

OSV integrates with various third-party tools to help you identify and mitigate vulnerabilities.  *These tools are community-built and not officially supported by OSV.*  Please refer to the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) for evaluating suitability.

Here are some examples:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)