[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV is a comprehensive, open-source database of vulnerabilities affecting open-source software, empowering developers to proactively identify and address security risks.**

This repository houses the infrastructure and code powering the OSV database and related services.

[View the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Comprehensive Vulnerability Data:** Access a vast and regularly updated database of known vulnerabilities.
*   **Go-based Scanner:** Utilize a powerful tool to scan your dependencies and identify vulnerabilities.
*   **Web UI:** Explore and search the OSV database through a user-friendly web interface.
*   **API Access:** Integrate OSV data directly into your tools and workflows via our API.
*   **Data Dumps:** Download data dumps for offline analysis and integration.
*   **Community Driven:** Benefit from a collaborative effort to improve vulnerability information.

## Core Components

This repository contains code for the following components:

*   `deployment/`: Infrastructure configuration (Terraform, Cloud Deploy, Cloud Build)
*   `docker/`: Dockerfiles for CI, deployment, and worker images.
*   `docs/`: Documentation files (Jekyll) and API documentation generation.
*   `gcp/api`: OSV API server code.
*   `gcp/datastore`: Datastore configuration.
*   `gcp/functions`: Cloud Functions for vulnerability import (e.g., PyPI).
*   `gcp/indexer`: Version determination and indexing tools.
*   `gcp/website`: Backend for the OSV web interface.
*   `gcp/workers/`: Background workers for tasks like bisection and impact analysis.
*   `osv/`: Core OSV Python library and ecosystem helpers.
*   `tools/`: Development scripts and utilities.
*   `vulnfeeds/`: Conversion tools for vulnerability feeds (e.g., NVD, Alpine, Debian).

## Getting Started

To build locally, you'll need to initialize submodules:

```bash
git submodule update --init --recursive
```

## Resources

*   **Documentation:** [Comprehensive documentation](https://google.github.io/osv.dev)
*   **API Documentation:** [API Documentation](https://google.github.io/osv.dev/api/)
*   **Data Dumps:** [Data dumps information](https://google.github.io/osv.dev/data/#data-dumps)
*   **Web UI:** <https://osv.dev>
*   **Scanner:** [OSV Scanner](https://github.com/google/osv-scanner)
*   **Mailing List:** [OSV Discuss](https://groups.google.com/g/osv-discuss)

## Contributing

We welcome contributions to OSV!  Learn more about contributing code, data, and documentation in the [CONTRIBUTING.md](CONTRIBUTING.md) file.  Have a question or suggestion? [Open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV data is integrated into various third-party tools.  *Note: These tools are community-built and not officially endorsed by OSV.*

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)