[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV.dev: Open Source Vulnerability Database

**OSV.dev is a powerful, open-source vulnerability database and API designed to improve the security of open-source software.**

This repository houses the infrastructure powering OSV.dev, providing a comprehensive solution for identifying and addressing vulnerabilities in your dependencies.

[Visit the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features:

*   **Comprehensive Vulnerability Database:** Access a vast and growing database of known vulnerabilities across various open-source projects.
*   **OSV API:** Integrate OSV.dev data into your tools and workflows with a robust and well-documented API.
*   **Dependency Scanning:** Utilize the OSV scanner ([https://github.com/google/osv-scanner](https://github.com/google/osv-scanner)) to identify vulnerabilities in your project's dependencies.
*   **Web UI:** Explore and search the OSV database through a user-friendly web interface available at <https://osv.dev>.
*   **Data Dumps:** Download data dumps for offline analysis and integration.
*   **Community Driven:** Benefit from an open and collaborative community dedicated to improving open-source security.

## Key Components of this Repository

This repository contains the following key components for running OSV.dev:

*   **Deployment:** Configuration files for Terraform and Cloud Deploy, along with Cloud Build configurations.
*   **Docker:** Docker files for CI, deployment, and worker base images.
*   **Documentation:** Jekyll files for the OSV.dev documentation.
*   **API:** Files for the OSV API server, including protobuf definitions.
*   **Data Storage:** Datastore index configuration.
*   **Web Interface:** Backend code for the OSV.dev web interface.
*   **Workers:** Background workers for bisection, impact analysis, and data processing.
*   **Core Library:** The core OSV Python library and ecosystem package versioning helpers.
*   **Tools:** Scripts for development and maintenance.
*   **Vulnerability Feeds:** Modules for converting vulnerability feeds from various sources.

## Getting Started

To build locally, you'll need to initialize submodules:

```bash
git submodule update --init --recursive
```

## Documentation

*   Comprehensive documentation is available [here](https://google.github.io/osv.dev).
*   API documentation is available [here](https://google.github.io/osv.dev/api/).
*   For more information about data dumps check out [our documentation](https://google.github.io/osv.dev/data/#data-dumps).

## Contribute

We welcome contributions! Learn more about contributing to [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation). You can also join our [mailing list](https://groups.google.com/g/osv-discuss) or [open an issue](https://github.com/google/osv.dev/issues) with questions or suggestions.

## Third-Party Tools and Integrations

OSV.dev is integrated with a variety of third-party tools. Note that these are community-built tools and are not supported or endorsed by the core OSV maintainers. Consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) to determine their suitability for your needs.

Some popular third-party tools are:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)