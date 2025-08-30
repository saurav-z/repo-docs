<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg">
</picture>

---

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database - Find & Fix Security Vulnerabilities

OSV (Open Source Vulnerability) is a free, open-source, and comprehensive vulnerability database that helps you identify and remediate security risks in your open-source dependencies. Developed by Google, OSV provides a centralized repository of vulnerability information, making it easier than ever to secure your software supply chain.  [Learn more on the OSV GitHub repository](https://github.com/google/osv.dev).

## Key Features of OSV

*   **Centralized Vulnerability Database:** A single source of truth for open-source vulnerabilities, simplifying security analysis.
*   **Dependency Scanning Tools:** Easily scan your projects and dependencies for known vulnerabilities using the OSV scanner.
*   **Comprehensive Coverage:** Supports a wide range of ecosystems and package managers, ensuring broad vulnerability detection.
*   **Data Dumps:** Access vulnerability data dumps for offline analysis and integration into your security workflows.
*   **Web UI:**  A user-friendly web interface for exploring vulnerabilities and staying informed.
*   **API Access:** Integrate OSV data directly into your security tools and processes.
*   **Community Driven:** Open to contributions from the community for data, code, and documentation.

## Key Components & Directories in this Repository

This repository powers the [OSV website](https://osv.dev) and API. Key directories include:

*   `deployment/`: Infrastructure as code configuration for cloud deployment.
*   `docker/`:  Dockerfiles for building CI and deployment images.
*   `docs/`:  Documentation files, including those used for the website.
*   `gcp/api`: OSV API server code, including protobuf definitions.
*   `gcp/datastore`: Datastore configuration.
*   `gcp/website`: Backend for the OSV web interface.
*   `gcp/workers/`: Background worker processes for data processing.
*   `osv/`: Core Python library and helper modules.
*   `vulnfeeds/`:  Tools for converting vulnerability feeds.

## Get Started with OSV

1.  **Explore the Web UI:** Browse vulnerabilities at <https://osv.dev>.
2.  **Use the Scanner:** Check your dependencies using the OSV scanner (available in its [own repository](https://github.com/google/osv-scanner)).
3.  **Access the API:** Integrate OSV data into your security tools via the [API documentation](https://google.github.io/osv.dev/api/).
4.  **Data Dumps:** Download vulnerability data from `gs://osv-vulnerabilities`.

## Contributing to OSV

We welcome contributions to help improve OSV!  Learn more about how you can contribute:

*   **Code:**  [CONTRIBUTING.md#contributing-code](CONTRIBUTING.md#contributing-code)
*   **Data:**  [CONTRIBUTING.md#contributing-data](CONTRIBUTING.md#contributing-data)
*   **Documentation:** [CONTRIBUTING.md#contributing-documentation](CONTRIBUTING.md#contributing-documentation)
*   **Questions and Suggestions:** Open an issue on [our issue tracker](https://github.com/google/osv.dev/issues).
*   **Mailing List:** Join our [mailing list](https://groups.google.com/g/osv-discuss).

## Third-Party Tools and Integrations

OSV is integrated into various security tools. Please consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) to determine suitability for your use. Popular integrations include:

*   Cortex XSOAR
*   dep-scan
*   Dependency-Track
*   GUAC
*   OSS Review Toolkit
*   pip-audit
*   Renovate
*   Trivy