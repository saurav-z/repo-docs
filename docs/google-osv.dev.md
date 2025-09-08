<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg" alt="OSV Logo">
</picture>

---

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: The Open Source Vulnerability Database

**OSV (Open Source Vulnerability) is a collaborative, open-source database of vulnerabilities in open-source software, designed to help developers identify and address security risks.**  Find out more and contribute on the [original repository](https://github.com/google/osv.dev).

## Key Features of OSV:

*   **Comprehensive Vulnerability Data:**  Access a vast database of known vulnerabilities affecting a wide range of open-source projects.
*   **API for Integration:**  Integrate OSV data directly into your security tools and workflows via a robust API.
*   **Dependency Scanning:** Scan your project's dependencies and check them against the OSV database for known vulnerabilities using the [OSV Scanner](https://github.com/google/osv-scanner).
*   **Web UI for Exploration:** Explore vulnerabilities, search for affected packages, and stay informed through the user-friendly web interface available at <https://osv.dev>.
*   **Data Dumps:** Access data dumps for offline analysis and integration with your systems.

## Explore the OSV Ecosystem

*   **Web UI:** <https://osv.dev>
*   **Documentation:** [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
*   **API Documentation:** [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)
*   **Data Dumps:** Accessible from a GCS bucket at `gs://osv-vulnerabilities` (see documentation for details)
*   **OSV Scanner:**  The [OSV Scanner](https://github.com/google/osv-scanner) helps you identify vulnerable dependencies.

## This Repository's Contents

This repository contains the code for running the OSV infrastructure on Google Cloud Platform (GCP).  Key directories include:

*   `deployment/`: Infrastructure-as-code configuration (Terraform, Cloud Deploy, Cloud Build).
*   `docker/`: Dockerfiles for CI, deployment, and worker images.
*   `docs/`: Documentation and website assets.
*   `gcp/api`:  OSV API server code.
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Functions.
*   `gcp/indexer`: Version determination indexer.
*   `gcp/website`: Backend of the OSV web interface.
*   `gcp/workers/`: Background workers for vulnerability analysis.
*   `osv/`: Core OSV Python library.
*   `tools/`: Utility scripts.
*   `vulnfeeds/`: Modules for vulnerability data conversion (e.g., NVD, Alpine, Debian).

### Submodule Initialization

To ensure proper functionality, initialize submodules:

```bash
git submodule update --init --recursive
```

## Contribute to OSV

OSV thrives on community contributions!  Learn more about:

*   [Contributing Code](CONTRIBUTING.md#contributing-code)
*   [Contributing Data](CONTRIBUTING.md#contributing-data)
*   [Contributing Documentation](CONTRIBUTING.md#contributing-documentation)
*   [OSV Mailing List](https://groups.google.com/g/osv-discuss)
*   [Report an Issue](https://github.com/google/osv.dev/issues)

## Third-Party Tools and Integrations

OSV is integrated into a number of third party tools. Please note that these community built tools are not supported or endorsed by OSV maintainers:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

*Please consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) to determine suitability for your use.*