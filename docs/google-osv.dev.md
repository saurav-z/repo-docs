<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg" alt="OSV Logo">
</picture>

---

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV is a comprehensive, open-source database and API for vulnerability information, helping developers identify and remediate security threats in their open-source dependencies.**  Explore the power of OSV to enhance your software security posture.

## Key Features

*   **Comprehensive Vulnerability Database:**  Access a centralized repository of known vulnerabilities affecting open-source software.
*   **API Access:** Integrate OSV data directly into your security tools and workflows through a robust API.
*   **Dependency Scanning:**  Utilize the provided scanner to identify vulnerabilities in your project's dependencies, supporting various lockfiles, SBOMs, and Git repositories.  See the [OSV Scanner](https://github.com/google/osv-scanner) repository.
*   **Web UI:**  Easily explore the OSV database and view vulnerability details through a user-friendly web interface at <https://osv.dev>.
*   **Data Dumps:** Access OSV vulnerability data through downloadable data dumps for offline analysis and integration.

## Key Components of This Repository

This repository houses the code and infrastructure that powers the OSV project, including:

*   **Deployment:** Terraform and Cloud Deploy configurations for infrastructure management.
*   **API Server:** Backend services for the OSV API.
*   **Data Indexing:**  Tools for indexing and organizing vulnerability information.
*   **Web Interface:**  The backend code for the OSV website.
*   **Worker Services:** Background processes for tasks like bisection and impact analysis.
*   **Core Libraries:**  The OSV Python library with core functionality.

## Getting Started

*   **Documentation:**  Dive deeper into the OSV project with comprehensive documentation available [here](https://google.github.io/osv.dev).
*   **API Documentation:**  Learn how to leverage the OSV API [here](https://google.github.io/osv.dev/api/).
*   **Data Dumps:** Access vulnerability data dumps from a GCS bucket at `gs://osv-vulnerabilities`. For more information, check out the [data documentation](https://google.github.io/osv.dev/data/#data-dumps).
*   **Submodules:** For local building, run `git submodule update --init --recursive`.

## Contribute to OSV

We welcome contributions to improve the OSV project. Learn more about contributing:

*   **Code:**  [CONTRIBUTING.md#contributing-code](CONTRIBUTING.md#contributing-code)
*   **Data:**  [CONTRIBUTING.md#contributing-data](CONTRIBUTING.md#contributing-data)
*   **Documentation:** [CONTRIBUTING.md#contributing-documentation](CONTRIBUTING.md#contributing-documentation)
*   **Mailing List:**  Join the discussion at [OSV Discuss](https://groups.google.com/g/osv-discuss).
*   **Open an Issue:**  Have questions or suggestions?  [Open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools & Integrations

OSV is supported by a vibrant community. Here are some third-party tools and integrations that use OSV data:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

---

**Visit the [OSV GitHub Repository](https://github.com/google/osv.dev) for more information and to get involved!**