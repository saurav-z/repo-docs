<!-- OSV.dev Logo -->
<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg" alt="OSV.dev Logo">
</picture>

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV.dev: Open Source Vulnerability Database & Tools

**OSV.dev is a comprehensive open-source vulnerability database and associated tools, helping developers identify and address security vulnerabilities in their software dependencies.** [Explore the OSV.dev repository](https://github.com/google/osv.dev) to learn more and contribute.

## Key Features of OSV.dev:

*   **Centralized Vulnerability Database:** Access a curated and up-to-date database of open-source vulnerabilities.
*   **Dependency Scanning:** Scan your project dependencies and check them against the OSV database for known vulnerabilities.
*   **API Access:** Leverage a robust API to integrate OSV data and functionality into your security workflows.
*   **Web UI:**  Browse the OSV database and explore vulnerability information through a user-friendly web interface at [https://osv.dev](https://osv.dev).
*   **Data Dumps:** Access vulnerability data dumps for offline analysis and integration with other tools (available on [https://google.github.io/osv.dev/data/#data-dumps](https://google.github.io/osv.dev/data/#data-dumps)).

## Using the OSV Scanner

The OSV project offers a Go-based scanner tool to identify vulnerabilities in your projects.  The scanner supports:

*   Various lockfiles (e.g., `package-lock.json`, `yarn.lock`, etc.)
*   Debian Docker containers
*   SPDX and CycloneDB SBOMs
*   Git repositories

Find the scanner in its dedicated [repository](https://github.com/google/osv-scanner).

## This Repository: Project Structure

This repository contains the core code and infrastructure for OSV.dev, including:

*   **Deployment configurations:**  Terraform & Cloud Deploy configurations.
*   **Docker files:**  CI and worker images.
*   **Documentation:**  Jekyll-based documentation for the OSV.dev website.
*   **API Server:**  OSV API server files (gcp/api).
*   **Data Processing:**  Indexer, worker, and importer components for data management.
*   **Web Interface Backend:** Backend for the [osv.dev](https://osv.dev) web interface.
*   **Core Library:** The `osv/` directory contains the core OSV Python library.
*   **Vulnerability Feed Conversion:** Tools for converting feeds from various sources like NVD and Alpine/Debian.

**Note:** To build locally, ensure you initialize submodules: `git submodule update --init --recursive`.

## Contributing

We welcome contributions to OSV.dev! Learn more about:

*   [Contributing Code](CONTRIBUTING.md#contributing-code)
*   [Contributing Data](CONTRIBUTING.md#contributing-data)
*   [Contributing Documentation](CONTRIBUTING.md#contributing-documentation)

Join the conversation on our [mailing list](https://groups.google.com/g/osv-discuss).

Have questions or suggestions? Please [open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV.dev is a valuable resource for many security tools.  Note that these are community-built tools and are not supported or endorsed by the core OSV maintainers.  Some examples include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

<!--
## Documentation
*   Comprehensive documentation is available [here](https://google.github.io/osv.dev).
*   API documentation is available [here](https://google.github.io/osv.dev/api/).
-->