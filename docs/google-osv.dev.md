[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**Stay ahead of security threats with OSV, a comprehensive database for open-source vulnerabilities.**  This repository houses the code that powers the OSV database, website, and API.

**[View the original repository on GitHub](https://github.com/google/osv.dev)**

## Key Features

*   **Comprehensive Vulnerability Database:** Access a constantly updated database of known vulnerabilities affecting open-source software.
*   **Powerful API:** Integrate vulnerability data directly into your security tools and workflows.  API documentation is available [here](https://google.github.io/osv.dev/api/).
*   **Web UI:** Browse and search vulnerabilities easily through the OSV web interface.  View the web UI at <https://osv.dev>.
*   **Dependency Scanning:** Utilize the OSV scanner, a Go-based tool, to scan your dependencies and identify vulnerabilities. More details are available in [its own repository](https://github.com/google/osv-scanner).
*   **Data Dumps:** Access data dumps for offline analysis and integration. Data dump information is available in the documentation [here](https://google.github.io/osv.dev/data/#data-dumps).
*   **Community-Driven:** Benefit from a community-supported project with contributions welcome from all.

## Repository Structure

This repository is the backbone of the OSV project and includes:

*   `deployment/`: Infrastructure configuration using Terraform & Cloud Deploy.
*   `docker/`: Dockerfiles for CI and worker image creation.
*   `docs/`: Documentation using Jekyll.
*   `gcp/api`:  OSV API server code and protobuf definitions.
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Functions for vulnerability ingestion (e.g. PyPI).
*   `gcp/indexer`: Version indexing service.
*   `gcp/website`: Backend code for the OSV web interface.
*   `gcp/workers/`: Background workers for various tasks (bisection, impact analysis).
*   `osv/`: Core OSV Python library and ecosystem helpers.
*   `tools/`: Development scripts and tools.
*   `vulnfeeds/`:  Vulnerability feed converters (NVD, Alpine, Debian).

**To build locally, you'll likely need to update submodules:**

```bash
git submodule update --init --recursive
```

## Contributing

We encourage contributions!  Learn more about contributing [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation).
Join the conversation on our [mailing list](https://groups.google.com/g/osv-discuss).

## Questions & Suggestions

[Open an issue](https://github.com/google/osv.dev/issues) to ask questions or provide feedback.

## Third-Party Tools & Integrations

OSV integrates with many popular tools.  Please consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) to determine suitability for your use. Examples include:

*   Cortex XSOAR
*   dep-scan
*   Dependency-Track
*   GUAC
*   OSS Review Toolkit
*   pip-audit
*   Renovate
*   Trivy