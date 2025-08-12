[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV provides a comprehensive and open database of known vulnerabilities affecting open-source software, empowering developers to secure their projects.** This repository houses the code that powers the OSV platform. For more information, visit the [OSV documentation](https://google.github.io/osv.dev/).

## Key Features

*   **Centralized Vulnerability Data:** Access a unified database of known vulnerabilities across various open-source projects.
*   **OSV API:** Utilize the OSV API to integrate vulnerability checks into your development workflows and CI/CD pipelines.
*   **Dependency Scanning:** Leverage the OSV scanner, available in its [own repository](https://github.com/google/osv-scanner), to identify vulnerable dependencies in your projects.
*   **Web UI:** Browse and search vulnerabilities through the OSV web interface, accessible at <https://osv.dev>.
*   **Data Dumps:** Access data dumps from a GCS bucket (`gs://osv-vulnerabilities`) for offline analysis and integration with other security tools.
*   **Community Driven:** Benefit from a community-driven project with opportunities for contribution.

## Repository Structure

This repository contains the core infrastructure and code for the OSV platform, including:

*   `deployment/`: Terraform and Cloud Deploy configuration files.
*   `docker/`: CI Docker files and worker images.
*   `docs/`: Jekyll files for the documentation site.
*   `gcp/api`: OSV API server files.
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Functions for publishing PyPI vulnerabilities.
*   `gcp/indexer`: Version determination indexer.
*   `gcp/website`: Backend for the OSV web interface.
*   `gcp/workers/`: Background workers for vulnerability analysis.
*   `osv/`: Core OSV Python library and ecosystem helpers.
*   `tools/`: Development scripts and utilities.
*   `vulnfeeds/`: Modules and converters for vulnerability feeds.

## Getting Started

To build locally, you may need to initialize submodules:

```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions!  Please refer to the following for contributing:

*   [Code](CONTRIBUTING.md#contributing-code)
*   [Data](CONTRIBUTING.md#contributing-data)
*   [Documentation](CONTRIBUTING.md#contributing-documentation)

Join the discussion on our [mailing list](https://groups.google.com/g/osv-discuss) or [open an issue](https://github.com/google/osv.dev/issues) with questions and suggestions.

## Third-Party Tools & Integrations

OSV integrates with a variety of third-party tools, including:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

*Note: These are community-built tools and are not officially supported by the OSV maintainers.*

---

**[Visit the OSV GitHub Repository](https://github.com/google/osv.dev) to learn more and get involved.**