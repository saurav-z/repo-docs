[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV is a comprehensive, free, and open-source vulnerability database and associated tooling, designed to help you identify and mitigate security vulnerabilities in your open-source software dependencies.**

[Visit the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Comprehensive Vulnerability Database:**  A curated and standardized database of open-source vulnerabilities, improving accuracy and ease of use.
*   **OSV API:** A robust API for querying and integrating OSV data into your security workflows.  API documentation is available [here](https://google.github.io/osv.dev/api/).
*   **Web UI:**  Browse and explore the OSV database through a user-friendly web interface at <https://osv.dev>.
*   **Dependency Scanning Tool:**  A Go-based scanner to check your project's dependencies against the OSV database, available in its [own repository](https://github.com/google/osv-scanner).
*   **Data Dumps:** Access raw vulnerability data for offline analysis and custom integrations. Data dumps are available from a GCS bucket at `gs://osv-vulnerabilities`. See the documentation for more information: [https://google.github.io/osv.dev/data/#data-dumps](https://google.github.io/osv.dev/data/#data-dumps)

## Repository Structure

This repository houses the core infrastructure for the OSV project, including:

*   **Deployment Configuration:**  Terraform and Cloud Deploy configuration files.
*   **Docker Images:**  CI and worker-related Docker files.
*   **Documentation:**  Jekyll files for the OSV documentation.
*   **API Server:**  OSV API server files, including protobuf definitions.
*   **Datastore:**  Datastore index configuration.
*   **Workers:** Background workers for various tasks like bisection and impact analysis.
*   **Core Python Library:** The central OSV Python library used across services.
*   **Vulnerability Feed Converters:** Tools to convert vulnerability data from various sources like NVD, Alpine, and Debian.

To build locally, ensure you update the submodules:

```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions! Learn more about contributing [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation).  Join the discussion on our [mailing list](https://groups.google.com/g/osv-discuss).  Have a question or suggestion?  [Open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools & Integrations

OSV is used by a growing ecosystem of security tools. Here are some popular third-party tools and integrations (not supported or endorsed by the OSV maintainers):

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)