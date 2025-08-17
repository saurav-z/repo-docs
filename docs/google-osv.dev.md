[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV is a comprehensive, open-source database that tracks and provides information about vulnerabilities in open-source software.**

## Key Features

*   **Centralized Vulnerability Data:** Provides a single source of truth for open-source vulnerability information.
*   **API Access:** Offers a robust API for querying and integrating vulnerability data into your security workflows.
*   **Dependency Scanning Tool:**  Includes a Go-based scanner to identify vulnerabilities in your project's dependencies.
*   **Web UI:**  A user-friendly web interface at [https://osv.dev](https://osv.dev) for browsing and searching vulnerabilities.
*   **Data Dumps:** Provides data dumps for offline analysis and integration.
*   **Community-Driven:**  Actively encourages contributions to improve data accuracy and coverage.

## Documentation

Comprehensive documentation is available at: [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
API documentation is available at: [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)
For more information about Data Dumps check out: [https://google.github.io/osv.dev/data/#data-dumps](https://google.github.io/osv.dev/data/#data-dumps)

## Using the Scanner

The Go-based OSV scanner helps you identify vulnerabilities in your project dependencies by scanning various lockfiles, Debian docker containers, SPDX and CycloneDB SBOMs, and Git repositories.

Find the scanner in its own repository: [https://github.com/google/osv-scanner](https://github.com/google/osv-scanner).

## Contributing

We welcome contributions! Learn more about:
*   [Code Contributions](CONTRIBUTING.md#contributing-code)
*   [Data Contributions](CONTRIBUTING.md#contributing-data)
*   [Documentation Contributions](CONTRIBUTING.md#contributing-documentation)
    
Join the discussion on our [mailing list](https://groups.google.com/g/osv-discuss) or [open an issue](https://github.com/google/osv.dev/issues) to ask questions or make suggestions.

## Third-Party Tools and Integrations

OSV integrates with several third-party tools and services. Please note that these are community-built and not officially supported or endorsed by OSV maintainers. We recommend consulting the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) before use.

Some tools that integrate with OSV:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

## This Repository

This repository contains the code for running the OSV platform on Google Cloud Platform (GCP), including deployment configurations, API server, data processing workers, and the web interface.

*   **Deployment:** `deployment/` - Terraform & Cloud Deploy config files, and Cloud Build config files.
*   **CI:** `docker/` - CI docker files, and `worker-base` docker image.
*   **Documentation:** `docs/` - Jekyll files for the OSV documentation.
*   **API:** `gcp/api` - OSV API server files and protobuf files.
*   **Datastore:** `gcp/datastore` - The datastore index file (`index.yaml`).
*   **Cloud Functions:** `gcp/functions` - Cloud Function for publishing PyPI vulnerabilities.
*   **Indexer:** `gcp/indexer` - The determine version `indexer`.
*   **Website Backend:** `gcp/website` - The backend of the osv.dev web interface.
*   **Workers:** `gcp/workers/` - Workers for bisection and impact analysis.
*   **Core Library:** `osv/` - The core OSV Python library.
*   **Tools:** `tools/` - Misc scripts/tools.
*   **Vulnerability Feeds:** `vulnfeeds/` - Go module for NVD CVE conversion, and Alpine/Debian feed converters.

To build locally, be sure to check out submodules:

```bash
git submodule update --init --recursive
```

---

**[Explore the OSV project on GitHub](https://github.com/google/osv.dev)**