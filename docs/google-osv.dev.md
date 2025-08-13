# OSV.dev: Your Comprehensive Vulnerability Database

**OSV.dev is an open-source vulnerability database and API designed to help you identify and mitigate security risks in your software supply chain.**

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

## Key Features

*   **Comprehensive Vulnerability Data:** Access a constantly updated database of known vulnerabilities affecting open-source software.
*   **Open Source:**  OSV.dev is open-source and community-driven, with contributions welcome.
*   **API Access:**  Integrate OSV data directly into your security tools and workflows using the OSV API.
*   **Web UI:** Easily browse and search the OSV database through the user-friendly web interface at <https://osv.dev>.
*   **Dependency Scanning:**  Utilize the OSV scanner ([https://github.com/google/osv-scanner](https://github.com/google/osv-scanner)) to scan your project's dependencies for known vulnerabilities.

## Explore the OSV Ecosystem

*   **Documentation:** Comprehensive documentation is available [here](https://google.github.io/osv.dev).
*   **API Documentation:**  API documentation is available [here](https://google.github.io/osv.dev/api/).
*   **Data Dumps:** Access data dumps from a GCS bucket at `gs://osv-vulnerabilities`. More information is available in the [documentation](https://google.github.io/osv.dev/data/#data-dumps).

## Repository Structure

This repository ([https://github.com/google/osv.dev](https://github.com/google/osv.dev)) contains the source code for running the OSV.dev platform on GCP, including:

*   `deployment/`:  Terraform & Cloud Deploy configuration.
*   `docker/`:  CI docker files and base images.
*   `docs/`:  Jekyll files for the OSV documentation.
*   `gcp/api`:  OSV API server files.
*   `gcp/datastore`:  Datastore index files.
*   `gcp/functions`:  Cloud Functions for PyPI vulnerability publishing.
*   `gcp/indexer`:  Version determination indexer.
*   `gcp/website`:  The backend for the OSV web interface.
*   `gcp/workers/`:  Workers for bisection, impact analysis, and database tasks.
*   `osv/`:  The core OSV Python library and ecosystem helpers.
*   `tools/`:  Development scripts and tools.
*   `vulnfeeds/`:  Modules for converting vulnerability feeds.

**Important:** For local building, ensure you initialize submodules:
```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions! Learn more about contributing [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation).  Join the discussion on the [mailing list](https://groups.google.com/g/osv-discuss).  For questions or suggestions, please [open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV.dev is integrated with many third-party tools. Please consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) to determine suitability for your use. Some popular tools include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)