[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database and Tools

**OSV (Open Source Vulnerability) is a database and set of tools for tracking and understanding vulnerabilities in open-source software.** This repository houses the infrastructure that powers the OSV platform. [Explore the original OSV repository](https://github.com/google/osv.dev).

## Key Features

*   **Comprehensive Vulnerability Database:** Access a vast database of known vulnerabilities affecting open-source projects.
*   **Vulnerability Scanning:**  Use the Go-based scanner (available in its [own repository](https://github.com/google/osv-scanner)) to scan your dependencies and identify potential vulnerabilities. The scanner can scan various lockfiles, Debian docker containers, SBOMs (SPDX and CycloneDB), and Git repositories.
*   **Web UI:**  Browse the OSV database and explore vulnerabilities via the web UI at <https://osv.dev>.
*   **Data Dumps:** Download data dumps from a GCS bucket for offline analysis and integration. Access data dumps at `gs://osv-vulnerabilities`.
*   **API Access:** Integrate with the OSV data via the OSV API.

## Repository Structure

This repository contains the code and configuration for running the OSV platform on Google Cloud Platform (GCP). Key directories and their purposes include:

*   `deployment/`: Terraform and Cloud Deploy configuration files.
*   `docker/`:  Dockerfiles for CI and deployment.
*   `docs/`:  Documentation files using Jekyll for https://google.github.io/osv.dev/.
*   `gcp/api`: OSV API server files.
*   `gcp/datastore`: Datastore index files.
*   `gcp/functions`: Cloud Functions for publishing PyPI vulnerabilities.
*   `gcp/indexer`: Version determination indexer.
*   `gcp/website`: Backend of the OSV web interface.
*   `gcp/workers/`:  Workers for various tasks (bisection, impact analysis, etc.).
*   `osv/`:  Core OSV Python library and ecosystem helpers.
*   `tools/`: Development and utility scripts.
*   `vulnfeeds/`:  Go module for vulnerability feed conversions (NVD, Alpine, Debian).

**Important:** Ensure you initialize submodules for local building to work correctly:

```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions! Learn more about contributing code, data, and documentation in the [CONTRIBUTING.md](https://github.com/google/osv.dev/blob/main/CONTRIBUTING.md) file.
Join the discussion on our [mailing list](https://groups.google.com/g/osv-discuss).  Please [open an issue](https://github.com/google/osv.dev/issues) for questions or suggestions.

## Third-Party Tools and Integrations

OSV integrates with several third-party tools.  Note these are community-built and not officially supported by OSV maintainers.  Consider the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) before use.

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)