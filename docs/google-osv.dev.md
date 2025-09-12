<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg" alt="OSV.dev Logo">
</picture>

---

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV.dev: Open Source Vulnerability Database

**OSV.dev is a comprehensive, open-source vulnerability database that helps you identify and address security risks in your software dependencies.** ([View the original repository](https://github.com/google/osv.dev))

## Key Features

*   **Centralized Vulnerability Data:** Access a curated database of known vulnerabilities affecting open-source software.
*   **Dependency Scanning:** Identify vulnerabilities in your projects through integration with tools like the OSV scanner.
*   **API Access:** Integrate OSV data into your security workflows with a robust API.
*   **Data Dumps:** Download data dumps for offline analysis and integration.
*   **Web UI:** Browse and search the OSV database through a user-friendly web interface at [https://osv.dev](https://osv.dev).
*   **Community Contributions:** Benefit from a community-driven database and contribute to its growth.

## Documentation & Resources

*   **Comprehensive Documentation:** Detailed documentation is available [here](https://google.github.io/osv.dev).
*   **API Documentation:** Explore the OSV API [here](https://google.github.io/osv.dev/api/).
*   **Data Dumps:** Access data dumps from a GCS bucket at `gs://osv-vulnerabilities`. Learn more [here](https://google.github.io/osv.dev/data/#data-dumps).
*   **OSV Scanner:** Scan your dependencies for vulnerabilities using the [OSV scanner](https://github.com/google/osv-scanner).

## This Repository Overview

This repository contains the code and infrastructure for running [https://osv.dev](https://osv.dev) on Google Cloud Platform (GCP). Key directories include:

*   `bindings/`: Language bindings for the OSV API (currently Go only).
*   `deployment/`: Terraform & Cloud Deploy configuration files.
*   `docker/`: CI and deployment Docker files.
*   `docs/`: Jekyll files for the OSV documentation.
*   `gcp/api`: OSV API server files.
*   `gcp/datastore`: Datastore index file.
*   `gcp/functions`: Cloud Functions (e.g., for PyPI vulnerability publishing).
*   `gcp/indexer`: Version indexing functionality.
*   `gcp/website`: Backend for the OSV.dev web interface.
*   `gcp/workers/`: Background workers for bisection, impact analysis, and other tasks.
*   `osv/`: The core OSV Python library.
*   `tools/`: Utility scripts and tools for development.
*   `vulnfeeds/`: Go module for vulnerability feed conversions (e.g., NVD, Alpine, Debian).

To build locally, initialize submodules:

```bash
git submodule update --init --recursive
```

## Contributing

Contributions are welcome!  Review the [CONTRIBUTING.md](https://github.com/google/osv.dev/blob/main/CONTRIBUTING.md) to learn how to contribute:

*   [Contributing Code](CONTRIBUTING.md#contributing-code)
*   [Contributing Data](CONTRIBUTING.md#contributing-data)
*   [Contributing Documentation](CONTRIBUTING.md#contributing-documentation)

Join the conversation on the [mailing list](https://groups.google.com/g/osv-discuss).  Have a question or suggestion?  Please [open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV.dev integrates with a number of third-party tools, enhancing its capabilities.  These tools are community-built and not officially supported or endorsed by the OSV maintainers.  See the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) for suitability.

Some popular integrations include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)