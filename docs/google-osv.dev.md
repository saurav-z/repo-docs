[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV.dev: Open Source Vulnerability Database

**OSV.dev is a comprehensive open-source vulnerability database and platform designed to help developers identify and address security issues in their software dependencies.**  This repository contains the code that powers the OSV.dev platform, providing a central hub for vulnerability information and tools to help you secure your open-source projects. ([Original Repository](https://github.com/google/osv.dev))

## Key Features

*   **Comprehensive Vulnerability Database:** Access a vast and growing database of known vulnerabilities affecting open-source software.
*   **API Access:** Integrate OSV data directly into your security tools and workflows using the OSV API ([API documentation](https://google.github.io/osv.dev/api/)).
*   **Web UI:** Easily browse and search the OSV database through the user-friendly web interface: <https://osv.dev>.
*   **Data Dumps:** Download data dumps for offline analysis and integration with your own security systems ([Data dump documentation](https://google.github.io/osv.dev/data/#data-dumps)).
*   **Dependency Scanning Tool:** Utilize the Go-based OSV scanner ([scanner repository](https://github.com/google/osv-scanner)) to identify vulnerable dependencies in your projects.  This tool supports various formats, including lockfiles, Debian containers, SPDX, CycloneDB SBOMs, and Git repositories.

## Repository Structure

This repository contains the infrastructure and code for running the OSV.dev platform on Google Cloud Platform (GCP), organized as follows:

*   `deployment/`: Terraform & Cloud Deploy configurations.
*   `docker/`: Dockerfiles for CI and deployment.
*   `docs/`: Documentation files (Jekyll for https://google.github.io/osv.dev/).
*   `gcp/api`: OSV API server files (including local ESP server).
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Function for PyPI vulnerability publishing.
*   `gcp/indexer`: Version determination indexer.
*   `gcp/website`: Backend for the OSV.dev web interface.
*   `gcp/workers/`: Background worker processes for various tasks.
*   `osv/`: Core OSV Python library.
*   `tools/`: Development scripts and tools.
*   `vulnfeeds/`: Go module for NVD CVE conversion.

## Getting Started

To build locally, you may need to check out submodules:

```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions from the community!  Learn more about contributing code, data, and documentation in our [CONTRIBUTING.md](CONTRIBUTING.md) file.  Join the conversation on our [mailing list](https://groups.google.com/g/osv-discuss).

Have questions or suggestions? [Open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV is compatible with a wide range of community-developed tools. Here are some popular options (not supported or endorsed by OSV):

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)