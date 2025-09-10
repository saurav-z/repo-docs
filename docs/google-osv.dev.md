<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg">
</picture>

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV is a comprehensive, open-source vulnerability database designed to improve the security of open-source software.**

[View the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features of OSV

*   **Comprehensive Vulnerability Data:** A curated database of vulnerabilities affecting open-source projects.
*   **Efficient Scanning & Detection:**  The OSV scanner tool identifies vulnerabilities in your project dependencies.
*   **API Access:** Access the OSV database through a well-defined API for easy integration.
*   **Data Dumps:** Access data dumps from a Google Cloud Storage bucket for offline analysis.
*   **Web UI:**  A user-friendly web interface for browsing and exploring vulnerabilities at <https://osv.dev>.
*   **Community-Driven:** Contributions and suggestions are welcome to improve the database.

##  Explore the OSV Ecosystem

### Documentation
Comprehensive documentation is available at [https://google.github.io/osv.dev](https://google.github.io/osv.dev). API documentation is available at [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/).

### Data Dumps
Access data dumps from a GCS bucket at `gs://osv-vulnerabilities`. For more information, check out [our documentation](https://google.github.io/osv.dev/data/#data-dumps).

### Web UI
The OSV web UI is deployed at <https://osv.dev>.

### OSV Scanner
The OSV scanner is a Go-based tool that checks your project dependencies against the OSV database for known vulnerabilities via the OSV API.

The scanner is located in its [own repository](https://github.com/google/osv-scanner).

## This Repository: Project Structure

This repository contains the code and configuration for running the OSV project on Google Cloud Platform (GCP). Key directories include:

*   `bindings/`: Language bindings for the OSV API (currently Go only)
*   `deployment/`: Terraform & Cloud Deploy config files.
*   `docker/`: CI and deployment Docker files.
*   `docs/`: Documentation files.
*   `gcp/api`: OSV API server files.
*   `gcp/datastore`: Datastore index file.
*   `gcp/functions`: Cloud Function for publishing PyPI vulnerabilities.
*   `gcp/indexer`: Version indexing functionality.
*   `gcp/website`: The backend of the osv.dev web interface.
*   `gcp/workers/`: Background worker processes.
*   `osv/`: The core OSV Python library.
*   `tools/`: Utility scripts for development.
*   `vulnfeeds/`: Go module for vulnerability data conversion.

To build locally, ensure you run: `git submodule update --init --recursive`

## Contributing

We welcome contributions!

*   **Code:** Learn more about [code contributions](CONTRIBUTING.md#contributing-code).
*   **Data:** Learn more about [data contributions](CONTRIBUTING.md#contributing-data).
*   **Documentation:** Learn more about [documentation contributions](CONTRIBUTING.md#contributing-documentation).

Join the discussion on the [mailing list](https://groups.google.com/g/osv-discuss).

For questions or suggestions, please [open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

The following are community-built tools, not supported or endorsed by OSV, but that integrate with OSV:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)