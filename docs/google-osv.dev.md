# OSV.dev: Your Central Hub for Open Source Vulnerability Information

**OSV.dev** is a comprehensive platform providing open-source vulnerability data, designed to help you identify and address security risks in your software dependencies.  This repository contains the code for running the OSV.dev infrastructure on Google Cloud Platform (GCP).  [Visit the original repository on GitHub](https://github.com/google/osv.dev) for the source code and further information.

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

## Key Features

*   **Centralized Vulnerability Database:** Access a constantly updated and comprehensive database of known vulnerabilities in open-source projects.
*   **Dependency Scanning:** Leverage a Go-based scanner to identify vulnerable dependencies within your projects.
*   **Web UI:** Explore vulnerabilities and project data via the user-friendly web interface at <https://osv.dev>.
*   **API Access:** Programmatically access OSV data through the OSV API.
*   **Data Dumps:** Download bulk vulnerability data via Google Cloud Storage (GCS).
*   **Extensive Documentation:** Find detailed information and guides on using OSV.dev.
*   **Community-Driven:** Benefit from a vibrant community and contribute to the project.

## Project Structure

This repository is a complex project composed of the following main components:

*   `deployment/`: Infrastructure as Code (Terraform & Cloud Deploy)
*   `docker/`: Docker files for CI and deployment.
*   `docs/`: Documentation files (Jekyll).
*   `gcp/api`: OSV API server files.
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Functions.
*   `gcp/indexer`: Version indexing tools.
*   `gcp/website`: Backend for the OSV web interface.
*   `gcp/workers/`: Background workers for data processing and analysis.
*   `osv/`: Core OSV Python library and related utilities.
*   `tools/`: Development scripts and tools.
*   `vulnfeeds/`: Tools for converting vulnerability data from various sources.

## Contributing

We encourage contributions from the community!  Learn more about how to contribute to:

*   [Code](CONTRIBUTING.md#contributing-code)
*   [Data](CONTRIBUTING.md#contributing-data)
*   [Documentation](CONTRIBUTING.md#contributing-documentation)

Join the discussion on our [mailing list](https://groups.google.com/g/osv-discuss).

Have a question or suggestion?  Please [open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV.dev integrates with a variety of third-party tools. Note that these are community-built tools and are not supported or endorsed by the OSV maintainers.

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

## Getting Started

To build locally, you'll need to initialize the submodules:

```bash
git submodule update --init --recursive