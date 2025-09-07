<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg">
</picture>

---

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database and Ecosystem

**OSV (Open Source Vulnerability) is a free, open, and comprehensive vulnerability database that helps developers find and fix vulnerabilities in their open-source dependencies.** This repository powers the OSV ecosystem, providing tools and resources for vulnerability management. Visit the [original repository](https://github.com/google/osv.dev) for the source code and more information.

## Key Features

*   **Comprehensive Vulnerability Database:** OSV provides a centralized, standardized database of open-source vulnerabilities.
*   **Vulnerability Scanning:**  A Go-based scanner to detect vulnerabilities in your project's dependencies, supporting various formats like lockfiles, SBOMs, and Git repositories.
*   **API Access:**  A robust API for querying vulnerability data, enabling seamless integration with existing security tools and workflows.
*   **Web UI:**  A user-friendly web interface (<https://osv.dev>) to browse and explore vulnerability information.
*   **Data Dumps:**  Publicly available data dumps for bulk download and offline analysis, hosted on Google Cloud Storage.
*   **Community-Driven:**  Welcomes contributions of code, data, and documentation.  A mailing list ([https://groups.google.com/g/osv-discuss](https://groups.google.com/g/osv-discuss)) is available for discussion and support.

## Documentation

*   Comprehensive documentation: [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
*   API documentation: [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)
*   Data Dumps: [https://google.github.io/osv.dev/data/#data-dumps](https://google.github.io/osv.dev/data/#data-dumps)

## Repository Structure

This repository contains the infrastructure and code for the OSV platform, including:

*   `deployment/`: Deployment configurations.
*   `docker/`: Dockerfiles for CI and workers.
*   `docs/`: Documentation source files.
*   `gcp/api`: OSV API server.
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Functions for vulnerability ingestion.
*   `gcp/indexer`: Version indexing components.
*   `gcp/website`: Backend for the OSV web interface.
*   `gcp/workers/`: Background worker processes.
*   `osv/`: The core OSV Python library.
*   `tools/`: Development and utility scripts.
*   `vulnfeeds/`: Vulnerability data feed conversion tools.

**Important:**  Ensure you initialize submodules before building locally: `git submodule update --init --recursive`

## Contributing

Contributions are welcome!  Learn more about contributing:

*   Code: [CONTRIBUTING.md#contributing-code](CONTRIBUTING.md#contributing-code)
*   Data: [CONTRIBUTING.md#contributing-data](CONTRIBUTING.md#contributing-data)
*   Documentation: [CONTRIBUTING.md#contributing-documentation](CONTRIBUTING.md#contributing-documentation)

Have questions or suggestions? [Open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV integrates with numerous third-party tools, including:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

*Note: These tools are community-maintained and not officially supported by OSV.*