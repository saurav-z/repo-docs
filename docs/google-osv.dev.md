[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV.dev: Open Source Vulnerability Database

**OSV.dev is a free, open-source database and API designed to help developers identify and remediate vulnerabilities in their open-source dependencies.** Explore the codebase and contribute to a safer open-source ecosystem at [https://github.com/google/osv.dev](https://github.com/google/osv.dev).

## Key Features

*   **Comprehensive Vulnerability Data:** Access a curated database of known vulnerabilities affecting open-source packages.
*   **Open API:** Programmatically query the OSV database to integrate vulnerability information into your tools and workflows.
*   **Dependency Scanning:** Use the dedicated OSV scanner ([https://github.com/google/osv-scanner](https://github.com/google/osv-scanner)) to identify vulnerable dependencies in your projects.
*   **Web UI:** Explore the OSV database and search for vulnerabilities through an intuitive web interface available at [https://osv.dev](https://osv.dev).
*   **Data Dumps:** Access data dumps for offline analysis and integration.
*   **Community Driven:** Actively maintained and supported by a community of developers.

## Project Structure

This repository contains the core infrastructure for the OSV.dev project, including:

*   `deployment/`: Infrastructure-as-code configuration using Terraform and Cloud Deploy.
*   `docker/`: Dockerfiles for CI and deployment processes.
*   `docs/`: Documentation and API specifications (generated with Jekyll).
*   `gcp/api`: OSV API server code (including protobuf definitions).
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Functions for PyPI vulnerability ingestion.
*   `gcp/indexer`: Version determination and indexing components.
*   `gcp/website`: Backend code for the OSV.dev web interface.
*   `gcp/workers/`: Background worker processes for data processing and analysis.
*   `osv/`: The core Python library and data model definitions.
*   `tools/`: Various scripts and utilities for development and maintenance.
*   `vulnfeeds/`: Modules for converting vulnerability feeds (e.g., NVD, Debian).

## Getting Started

To build locally, ensure you initialize and update submodules:

```bash
git submodule update --init --recursive
```

## Documentation and Resources

*   **Comprehensive Documentation:** [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
*   **API Documentation:** [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)
*   **Data Dumps:** Available from `gs://osv-vulnerabilities`. Learn more in the [documentation](https://google.github.io/osv.dev/data/#data-dumps).
*   **Web UI:** <https://osv.dev>
*   **Scanner Repository:** [https://github.com/google/osv-scanner](https://github.com/google/osv-scanner)

## Contributing

We welcome contributions! Learn how to contribute code, data, and documentation in [CONTRIBUTING.md](CONTRIBUTING.md).
Join the conversation on our [mailing list](https://groups.google.com/g/osv-discuss).
Report issues or suggest improvements by [opening an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV.dev is integrated with many third-party tools. Please consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software)
to determine suitability for your use. Some tools using OSV include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)