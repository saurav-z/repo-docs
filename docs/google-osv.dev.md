[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV is a free, open-source, and community-driven vulnerability database designed to improve the security of open-source software.** This repository is the core infrastructure behind OSV.

[View the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Comprehensive Vulnerability Data:** OSV provides a centralized and standardized database of known vulnerabilities across various open-source projects.
*   **Free and Open Source:** Accessible to everyone, empowering the community to improve software security.
*   **Integration with Existing Tools:** OSV is compatible with tools like Trivy, Dependency-Track, and Renovate.
*   **Regular Updates:** The database is continuously updated with new vulnerabilities.
*   **API Access:** A robust API allows developers to query the OSV database for vulnerability information.

## Key Components of This Repository

This repository contains the source code and configuration for the OSV infrastructure, including:

*   **Deployment:** Terraform and Cloud Deploy configurations.
*   **API Server:** OSV API server files.
*   **Website Backend:** Code for the OSV web interface.
*   **Data Processing Workers:** Workers for bisection, impact analysis, importing, and exporting vulnerability data.
*   **Core OSV Library:** The OSV Python library, used across multiple services.
*   **Vulnerability Data Feeds:** Converters for various vulnerability data feeds.

## Getting Started

To build locally, update submodules:

```bash
git submodule update --init --recursive
```

## Useful Links

*   **Documentation:** [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
*   **API Documentation:** [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)
*   **Web UI:** <https://osv.dev>
*   **Data Dumps:** Available from a GCS bucket at `gs://osv-vulnerabilities`. More information in our [documentation](https://google.github.io/osv.dev/data/#data-dumps).
*   **OSV Scanner:** [https://github.com/google/osv-scanner](https://github.com/google/osv-scanner)

## Contributing

We welcome contributions! Learn how to contribute to [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation). Join the [mailing list](https://groups.google.com/g/osv-discuss) or [open an issue](https://github.com/google/osv.dev/issues) with questions or suggestions.

## Third-Party Tools and Integrations

OSV integrates with various community-built tools. Note these are not supported or endorsed by the core OSV maintainers.

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)