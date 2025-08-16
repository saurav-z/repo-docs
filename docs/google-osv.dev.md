[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database and Infrastructure

**OSV provides a comprehensive database and infrastructure for open-source vulnerability information, helping you proactively identify and mitigate security risks in your software supply chain.** This repository hosts the core components of the OSV platform, built and maintained by Google.

[Visit the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Comprehensive Vulnerability Data:** A centralized database of open-source vulnerabilities, providing a single source of truth for security information.
*   **Web UI:** Easily browse and search the OSV vulnerability database at [https://osv.dev](https://osv.dev).
*   **API Access:** Access vulnerability data programmatically via the OSV API, with detailed API documentation available [here](https://google.github.io/osv.dev/api/).
*   **Dependency Scanning Tool:** Scan your project dependencies and check for known vulnerabilities using the OSV scanner, available in its [own repository](https://github.com/google/osv-scanner).
*   **Data Dumps:** Download vulnerability data for offline analysis and integration via data dumps from a GCS bucket at `gs://osv-vulnerabilities`. See [our documentation](https://google.github.io/osv.dev/data/#data-dumps) for details.

## Repository Structure

This repository contains the code for running the OSV platform, including:

*   **Deployment:** Configuration files for Terraform and Cloud Deploy.
*   **Docker:** Dockerfiles for CI, deployment, and worker images.
*   **Documentation:** Jekyll files for the OSV documentation website ([https://google.github.io/osv.dev/](https://google.github.io/osv.dev/)).
*   **API Server:** Files for the OSV API server, including protobuf definitions.
*   **Datastore:** Datastore index configuration.
*   **Cloud Functions:** Cloud Functions for various tasks, including publishing PyPI vulnerabilities.
*   **Workers:** Background workers for tasks like bisection, impact analysis, and data import/export.
*   **OSV Core Library:** The core Python library used across OSV services.
*   **Vulnerability Feeds:** Modules for converting vulnerability data from various sources, such as the NVD and Debian.
*   **Tools:** Scripts and tools for development and maintenance.

To build locally, remember to initialize submodules:

```bash
git submodule update --init --recursive
```

## Contributing

Contributions are welcome! Learn how to contribute by reviewing the [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation) contribution guidelines.
Have a question or suggestion? Please [open an issue](https://github.com/google/osv.dev/issues). You can also join the [mailing list](https://groups.google.com/g/osv-discuss).

## Third-Party Tools and Integrations

Explore community-built tools and integrations that leverage OSV. Note that these tools are not supported or endorsed by the OSV maintainers. Some popular third-party tools are:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)