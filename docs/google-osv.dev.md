[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database and Infrastructure

**OSV (Open Source Vulnerability) is a comprehensive, community-driven database and infrastructure for open-source vulnerability information, helping you stay secure.**

[View the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features of OSV

*   **Comprehensive Vulnerability Database:** A central repository for open-source vulnerability data.
*   **OSV API:** Provides a robust API for querying and integrating vulnerability information into your security workflows.
*   **Dependency Scanning:** Integrate with the [OSV scanner](https://github.com/google/osv-scanner) to scan dependencies for vulnerabilities.
*   **Web UI:** Provides a web interface for exploring and searching the OSV database.
*   **Data Dumps:** Access data dumps in a GCS bucket for bulk data access and analysis.
*   **Community-Driven:** OSV is open for contributions and improvements.

## Core Components and Technologies

This repository powers the OSV infrastructure and contains code for various key functionalities:

*   **Deployment & Infrastructure:** Terraform and Cloud Deploy configuration for deploying OSV on GCP.
*   **API Server:** The OSV API server, built using Go and protobuf files.
*   **Web Interface:** The backend of the OSV web interface, with the frontend located in `gcp/website`.
*   **Data Processing Workers:** Workers for tasks such as bisection and impact analysis.
*   **OSV Python Library:** Core Python library used across various OSV services, including Datastore model definitions.
*   **Vulnerability Feed Converters:** Go module and tools for converting vulnerability feeds, including NVD CVE and others.

## Getting Started

To build locally, you'll need to initialize submodules:

```bash
git submodule update --init --recursive
```

## Resources

*   **Documentation:** [Comprehensive documentation](https://google.github.io/osv.dev)
*   **API Documentation:** [API documentation](https://google.github.io/osv.dev/api/)
*   **Data Dumps:** Access data dumps at `gs://osv-vulnerabilities` (see [documentation](https://google.github.io/osv.dev/data/#data-dumps))
*   **Web UI:** Explore the web UI at <https://osv.dev>

## Contributing

We encourage contributions!  Learn how to contribute:

*   [Code](CONTRIBUTING.md#contributing-code)
*   [Data](CONTRIBUTING.md#contributing-data)
*   [Documentation](CONTRIBUTING.md#contributing-documentation)
*   [Mailing List](https://groups.google.com/g/osv-discuss)
*   [Open an Issue](https://github.com/google/osv.dev/issues)

## Third-Party Tools and Integrations

OSV is compatible with several community-built tools. These are not supported or endorsed by the OSV maintainers. Check out the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) to determine suitability. Example tools:

*   Cortex XSOAR
*   dep-scan
*   Dependency-Track
*   GUAC
*   OSS Review Toolkit
*   pip-audit
*   Renovate
*   Trivy