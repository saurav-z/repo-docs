<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg">
</picture>

---

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database & Tooling

**OSV.dev provides a comprehensive, open-source vulnerability database and a suite of tools to help you identify and manage security vulnerabilities in your open-source dependencies.** Developed by Google, OSV.dev empowers developers and security professionals to stay ahead of potential threats.  Explore the project on [GitHub](https://github.com/google/osv.dev).

## Key Features of OSV.dev:

*   **Centralized Vulnerability Database:** Access a curated database of vulnerabilities affecting open-source software.
*   **Vulnerability Scanner:** Quickly identify vulnerable dependencies in your projects using the OSV scanner.
*   **API Access:** Integrate OSV data into your existing security workflows through a robust API.
*   **Web UI:** Easily browse vulnerabilities and explore the OSV database through the user-friendly web interface.
*   **Data Dumps:** Download vulnerability data for offline analysis and integration.
*   **Community Driven:** Benefit from contributions from the open-source community and actively participate in improving the database.

## Key Resources

*   **Documentation:** Comprehensive documentation is available [here](https://google.github.io/osv.dev).
*   **API Documentation:** Access the API documentation [here](https://google.github.io/osv.dev/api/).
*   **Web UI:** Explore the OSV web interface at <https://osv.dev>.
*   **Data Dumps:** Download data dumps from `gs://osv-vulnerabilities`. Learn more [here](https://google.github.io/osv.dev/data/#data-dumps).
*   **OSV Scanner:** The scanner is located in its [own repository](https://github.com/google/osv-scanner).

## Repository Structure

This repository contains the code for running the OSV platform on Google Cloud Platform (GCP). It includes the following key components:

*   `deployment/`: Terraform and Cloud Deploy configuration files.
*   `docker/`: CI docker files.
*   `docs/`: Jekyll files for the OSV documentation.
*   `gcp/api`: OSV API server files and protobuf definitions.
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Function for publishing PyPI vulnerabilities.
*   `gcp/indexer`: Version indexing tools.
*   `gcp/website`: Backend of the OSV web interface.
*   `gcp/workers/`: Workers for various tasks (bisection, import, export, alias, etc.).
*   `osv/`: Core OSV Python library and related helpers.
*   `tools/`: Development scripts and utilities.
*   `vulnfeeds/`: Vulnerability data conversion tools (NVD, Alpine, Debian).

**Note:** You'll need to initialize submodules for local building:
```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions!  Learn how to contribute [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation).
Join the discussion on the [mailing list](https://groups.google.com/g/osv-discuss).  Have a question or suggestion?  [Open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools & Integrations

OSV.dev is compatible with a variety of community-developed tools.  These tools are not supported or endorsed by the core OSV maintainers.
Some popular tools include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)