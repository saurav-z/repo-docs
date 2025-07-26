[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: The Open Source Vulnerability Database

**OSV is a comprehensive, open-source database of vulnerabilities affecting open-source software, providing valuable information for security researchers and developers.** Find out more about OSV at [https://github.com/google/osv.dev](https://github.com/google/osv.dev).

## Key Features

*   **Comprehensive Vulnerability Database:**  Centralized repository for known vulnerabilities, allowing for efficient vulnerability assessment.
*   **API Access:**  Provides a well-documented API to access and integrate OSV data into security tools and workflows. (API documentation available [here](https://google.github.io/osv.dev/api/))
*   **Data Dumps:**  Regular data dumps available for bulk access and offline analysis.  See [documentation](https://google.github.io/osv.dev/data/#data-dumps) for more details.
*   **Web UI:** User-friendly web interface for browsing and searching vulnerabilities at [https://osv.dev](https://osv.dev).
*   **Scanning Tool:**  Go-based scanner to check your project dependencies against the OSV database. (Scanner located in its [own repository](https://github.com/google/osv-scanner).)

## This Repository: Infrastructure for OSV

This repository contains the core infrastructure and code that powers the OSV website and API, built on Google Cloud Platform (GCP).  It includes:

*   Deployment configurations (Terraform, Cloud Deploy)
*   Docker configurations for CI/CD and worker images
*   Documentation files (Jekyll for the website)
*   OSV API server code and Protobuf definitions
*   Datastore index files
*   Backend code for the OSV web interface
*   Worker processes for vulnerability analysis
*   Core OSV Python library
*   Tools for development and maintenance
*   Vulnerability feed converters

To build locally, you'll need to initialize submodules:
```bash
git submodule update --init --recursive
```
## Contributing

Contributions are welcome! Learn more about contributing [here](CONTRIBUTING.md).
You can contribute by:

*   Submitting Code [CONTRIBUTING.md#contributing-code]
*   Contributing Data [CONTRIBUTING.md#contributing-data]
*   Contributing Documentation [CONTRIBUTING.md#contributing-documentation]

Have a question or suggestion? [Open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV is used by a vibrant community of security tools.  Here are some examples:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)