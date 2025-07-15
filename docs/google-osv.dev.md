[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV (Open Source Vulnerability) is a comprehensive database and ecosystem for open-source vulnerability information, helping you secure your software supply chain.**  This repository contains the code for the OSV project, hosted on Google Cloud Platform.  Explore the official OSV project on [GitHub](https://github.com/google/osv.dev).

## Key Features

*   **Comprehensive Vulnerability Data:** Access a centralized database of known vulnerabilities affecting open-source projects.
*   **API Access:** Integrate vulnerability data into your security tools and workflows via the OSV API.
*   **Web UI:** Easily browse and search for vulnerabilities at [https://osv.dev](https://osv.dev).
*   **Scanning Tools:** Use the provided Go-based scanner to identify vulnerabilities in your project dependencies.
*   **Data Dumps:** Download data dumps for offline analysis and integration with other systems.

## Project Structure

This repository contains the infrastructure and code for running the OSV project, including:

*   `deployment/`: Terraform and Cloud Deploy configuration for deployment on GCP.
*   `docker/`: Docker files for CI and deployment.
*   `docs/`: Documentation files for the OSV documentation site.
*   `gcp/api`: OSV API server implementation.
*   `gcp/datastore`: Datastore configuration.
*   `gcp/functions`: Cloud Functions for vulnerability ingestion.
*   `gcp/indexer`: Version determination and indexing tools.
*   `gcp/website`: Backend for the OSV web interface.
*   `gcp/workers/`: Background workers for various tasks.
*   `osv/`: The core OSV Python library and related components.
*   `tools/`: Development tools and scripts.
*   `vulnfeeds/`: Tools for converting vulnerability feeds.

**To build locally, you may need to update submodules:**

```bash
git submodule update --init --recursive
```

## Getting Started

*   **Documentation:** Comprehensive documentation is available at [https://google.github.io/osv.dev](https://google.github.io/osv.dev).
*   **API Documentation:** Explore the API at [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/).
*   **Data Dumps:** Access data dumps from a GCS bucket at `gs://osv-vulnerabilities`.  See the [documentation](https://google.github.io/osv.dev/data/#data-dumps) for more information.
*   **Scanner:**  The OSV scanner is available in its [own repository](https://github.com/google/osv-scanner).

## Contributing

Contributions are welcome! Please review our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on contributing code, data, and documentation.  Join the discussion on the [mailing list](https://groups.google.com/g/osv-discuss).  For questions or suggestions, please [open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

The OSV ecosystem is supported by various third-party tools and integrations:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

*Note: These tools are community-maintained and not officially supported by OSV.*