[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database & Infrastructure

**OSV (Open Source Vulnerability) provides a comprehensive, centralized database and infrastructure for open-source vulnerability information, empowering developers to identify and mitigate security risks effectively.** Explore the OSV project on [GitHub](https://github.com/google/osv.dev).

## Key Features

*   **Centralized Vulnerability Database:** A single source of truth for open-source vulnerabilities.
*   **Vulnerability Scanning Tool:** Detect vulnerabilities in your project dependencies using the [OSV scanner](https://github.com/google/osv-scanner).
*   **Web UI:** Easily browse and search for vulnerabilities through the [OSV web interface](https://osv.dev).
*   **API Access:** Integrate OSV data into your security tools and workflows via the [OSV API](https://google.github.io/osv.dev/api/).
*   **Data Dumps:** Access raw vulnerability data for offline analysis and integration via the [OSV data dumps](https://google.github.io/osv.dev/data/#data-dumps).
*   **Community-Driven:** A collaborative effort to improve and expand the database, with contributions welcomed.

## Project Structure

This repository contains the code for running and maintaining the OSV infrastructure on Google Cloud Platform (GCP). Key components include:

*   `deployment/`: Terraform and Cloud Deploy configuration.
*   `docker/`: Dockerfiles for CI and deployment.
*   `docs/`: Documentation files for the OSV website.
*   `gcp/api`: OSV API server files.
*   `gcp/datastore`: Datastore configuration.
*   `gcp/functions`: Cloud Functions for specific tasks (e.g., PyPI vulnerability updates).
*   `gcp/indexer`: Version indexing tools.
*   `gcp/website`: Backend for the OSV web interface.
*   `gcp/workers/`: Background workers for various tasks (bisection, import, export, alias).
*   `osv/`: Core OSV Python library.
*   `tools/`: Development scripts and utilities.
*   `vulnfeeds/`: Tools for vulnerability data conversion (e.g., NVD, Alpine, Debian).

**Note:** For local building, ensure you initialize submodules: `git submodule update --init --recursive`

## Contributing

We welcome contributions! Learn more about:

*   [Contributing Code](CONTRIBUTING.md#contributing-code)
*   [Contributing Data](CONTRIBUTING.md#contributing-data)
*   [Contributing Documentation](CONTRIBUTING.md#contributing-documentation)

Join the discussion on the [mailing list](https://groups.google.com/g/osv-discuss).

Have a question or suggestion? [Open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV data is used by several community-built tools.  Please refer to the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) to determine suitability for your use. Some popular tools include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)