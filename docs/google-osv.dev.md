<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg">
</picture>

---

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV (Open Source Vulnerability) is a comprehensive database and API for open-source vulnerability data, empowering developers to proactively identify and address security risks in their software dependencies.** Explore the [OSV repository](https://github.com/google/osv.dev) to learn more.

## Key Features

*   **Comprehensive Vulnerability Data:** Access a vast and constantly updated database of open-source vulnerabilities.
*   **API Access:** Integrate the OSV API into your security tools and workflows for automated vulnerability detection.
*   **Dependency Scanning:** Utilize the provided Go-based scanner to identify vulnerable dependencies in your projects.
*   **Web UI:**  Browse and explore vulnerabilities through the user-friendly OSV web interface (<https://osv.dev>).
*   **Data Dumps:** Access raw vulnerability data through Google Cloud Storage (GCS) for custom analysis and integration.
*   **Multiple Ecosystem Support:**  Supports scanning and vulnerability information for various package ecosystems.

## Key Components of This Repository

This repository houses the infrastructure and code that powers the OSV platform, including:

*   **Deployment Configuration:** Terraform and Cloud Deploy configuration files.
*   **Docker Images:** CI, deployment, and worker images for various processes.
*   **Documentation:** Jekyll files for the OSV documentation site.
*   **API Server:** The OSV API server files, including Protobuf definitions.
*   **Data Processing & Indexing:** Components for data indexing and processing.
*   **Web Interface Backend:**  The backend code for the OSV web interface.
*   **Worker Processes:** Background workers for tasks like bisection and impact analysis.
*   **Core OSV Library:**  The core OSV Python library and ecosystem helpers.
*   **Vulnerability Feed Converters:**  Tools for converting vulnerability feeds from sources like NVD, Alpine, and Debian.

## Getting Started

To build locally, you will need to initialize the submodules:

```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions to OSV!  Learn how to contribute code, data, and documentation by exploring the [CONTRIBUTING.md](CONTRIBUTING.md) guide. Join the community discussion on the [mailing list](https://groups.google.com/g/osv-discuss) and report issues [here](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV integrates with various third-party tools to enhance your security posture. Please consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) to determine suitability for your use. Here are some popular tools:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

## Resources

*   **Documentation:** [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
*   **API Documentation:** [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)
*   **Data Dumps:** `gs://osv-vulnerabilities`
*   **Scanner Repository:** [https://github.com/google/osv-scanner](https://github.com/google/osv-scanner)