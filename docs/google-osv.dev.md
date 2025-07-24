[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV (Open Source Vulnerability) is a comprehensive, open-source database designed to provide accurate and actionable vulnerability information for open-source software projects.** This repository houses the core code for running the OSV platform, enabling users and security professionals to identify and mitigate security risks in their software dependencies. [Visit the original repository on GitHub](https://github.com/google/osv.dev).

## Key Features

*   **Comprehensive Vulnerability Database:** A centralized repository of vulnerability data for open-source projects.
*   **API Access:**  Provides a robust API to query the OSV database and integrate vulnerability information into your security workflows.
*   **Web UI:** A user-friendly web interface (<https://osv.dev>) for browsing and searching for vulnerabilities.
*   **Dependency Scanning:** Includes a Go-based scanner ([OSV Scanner](https://github.com/google/osv-scanner)) to check your dependencies against the OSV database.
*   **Data Dumps:**  Availability of data dumps from a GCS bucket (`gs://osv-vulnerabilities`) for offline analysis and integration.
*   **Community Driven:** Actively welcomes contributions to improve the database and platform.

## Getting Started

### Documentation

*   Comprehensive documentation is available at: <https://google.github.io/osv.dev>
*   API documentation: <https://google.github.io/osv.dev/api/>

### Data Dumps

Access OSV data dumps for offline use: <https://google.github.io/osv.dev/data/#data-dumps>

### Web UI

Browse the OSV web interface at: <https://osv.dev>

### Using the Scanner

Utilize the Go-based OSV scanner tool to identify vulnerabilities in your dependencies:  [OSV Scanner](https://github.com/google/osv-scanner).

## Repository Structure

This repository contains the code for running the OSV platform on Google Cloud Platform (GCP). Key directories include:

*   `deployment/`: Terraform & Cloud Deploy configuration.
*   `docker/`: Docker files for CI and deployment.
*   `docs/`: Jekyll files for the OSV documentation.
*   `gcp/api`: OSV API server files.
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Functions (e.g., PyPI vulnerability publishing).
*   `gcp/indexer`: Version determination indexer.
*   `gcp/website`: Backend of the OSV web interface.
*   `gcp/workers/`: Background workers for various tasks (bisection, importing, exporting).
*   `osv/`: Core OSV Python library and related modules.
*   `tools/`: Development scripts and tools.
*   `vulnfeeds/`:  Modules for vulnerability data conversion (e.g., NVD, Alpine, Debian).

To build locally, make sure to initialize submodules:

```bash
git submodule update --init --recursive
```

## Contributing

Contributions are welcome!  Please refer to the following for contribution guidelines:

*   [Contributing Code](CONTRIBUTING.md#contributing-code)
*   [Contributing Data](CONTRIBUTING.md#contributing-data)
*   [Contributing Documentation](CONTRIBUTING.md#contributing-documentation)

For questions and suggestions, please [open an issue](https://github.com/google/osv.dev/issues).
For discussions and announcements, you can join the [mailing list](https://groups.google.com/g/osv-discuss).

## Third-Party Tools and Integrations

Explore community-built tools that integrate with OSV:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

*Note: These are community tools and are not supported or endorsed by the core OSV maintainers.*