[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database and Infrastructure

**OSV (Open Source Vulnerability) is a powerful, free, and open-source database and infrastructure designed to improve the security of open-source software.** This repository ([https://github.com/google/osv.dev](https://github.com/google/osv.dev)) provides the codebase for running the OSV platform on Google Cloud Platform (GCP).

## Key Features

*   **Comprehensive Vulnerability Database:** Access a centralized, high-quality database of known vulnerabilities affecting open-source software.
*   **Automated Scanning:** Utilize tools to scan your project's dependencies and identify vulnerabilities.
*   **API Access:** Integrate OSV data into your security workflows through a robust API.
*   **Web UI:** Easily search and explore vulnerabilities through a user-friendly web interface (<https://osv.dev>).
*   **Data Dumps:** Download data dumps for offline analysis and integration.
*   **Open Source:** Benefit from a community-driven, transparent, and auditable security resource.

## Core Components of this Repository

This repository houses the infrastructure that powers the OSV platform, including:

*   **Deployment Configuration:** Terraform and Cloud Deploy configurations for GCP deployment.
*   **Docker Images:** Dockerfiles for CI, deployment, and worker base images.
*   **Documentation:** Jekyll files for the OSV documentation website.
*   **API Server:** Code for the OSV API server and protobuf definitions.
*   **Data Processing:**  Code for indexing, worker processes for analysis, and data import/export.
*   **Web Interface:** Backend code for the OSV web interface.
*   **Core Library:** The core OSV Python library for vulnerability data handling and processing.
*   **Vulnerability Feeds:** Modules for converting vulnerability data from various sources (e.g., NVD, Alpine, Debian).

**Note:** For the OSV scanner tool, please see the separate repository: [https://github.com/google/osv-scanner](https://github.com/google/osv-scanner).

## Getting Started

To build locally, you will need to checkout the submodules:

```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions! Learn more about contributing to code, data, and documentation in the [CONTRIBUTING.md](CONTRIBUTING.md) file. Join the discussion on our [mailing list](https://groups.google.com/g/osv-discuss). Have a question or suggestion? Please [open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

A vibrant community supports OSV, building tools and integrations. Please consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) for suitability in your use. Some popular third-party tools include:

*   Cortex XSOAR
*   dep-scan
*   Dependency-Track
*   GUAC
*   OSS Review Toolkit
*   pip-audit
*   Renovate
*   Trivy