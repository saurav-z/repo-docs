[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV.dev: Your Open Source Vulnerability Database

**OSV.dev is a comprehensive, open-source vulnerability database and associated tooling to help you identify and mitigate security risks in your open-source dependencies.** This repository contains the code that powers the OSV.dev platform.

[View the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Comprehensive Vulnerability Data:** Access a curated database of known vulnerabilities affecting open-source software.
*   **Vulnerability Scanning:**  Scan your dependencies to identify potential vulnerabilities. Use the companion scanner tool, [osv-scanner](https://github.com/google/osv-scanner).
*   **Web UI:** Easily browse and search vulnerabilities at <https://osv.dev>.
*   **API Access:**  Integrate with the OSV database through a robust API. [API Documentation](https://google.github.io/osv.dev/api/)
*   **Data Dumps:** Download data dumps for offline analysis and integration. [Data Dump Documentation](https://google.github.io/osv.dev/data/#data-dumps)

## Repository Structure

This repository encompasses the backend infrastructure and codebase for OSV.dev, including:

*   `deployment/`:  Terraform and Cloud Deploy configuration files for deployment.
*   `docker/`: CI and deployment Docker files, plus base images.
*   `docs/`:  Jekyll-based documentation and related build scripts.
*   `gcp/api`:  OSV API server files.
*   `gcp/datastore`: Datastore configuration.
*   `gcp/functions`: Cloud Functions for PyPI vulnerability publishing.
*   `gcp/indexer`:  Version determination indexer.
*   `gcp/website`:  The backend for the OSV.dev web interface, with the frontend in `frontend3`.
*   `gcp/workers/`:  Background workers for tasks such as bisection and impact analysis.
*   `osv/`:  Core OSV Python library.
*   `tools/`:  Development scripts and utilities.
*   `vulnfeeds/`:  Go module for vulnerability data conversion (e.g., NVD CVE).

**Important:** You'll need to initialize submodules for local building:

```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions from the community!  Learn how to contribute to [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation).

*   **Mailing List:** [OSV-Discuss](https://groups.google.com/g/osv-discuss)
*   **Issues:** [Open an issue](https://github.com/google/osv.dev/issues)

## Third-Party Tools and Integrations

OSV.dev is supported by a vibrant ecosystem of third-party tools and integrations.  Please note these are community-built and not officially supported by OSV.

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)