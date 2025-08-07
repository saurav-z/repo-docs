[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV is a community-driven, open-source vulnerability database that provides a unified and comprehensive view of security vulnerabilities in open-source software.**  This repository houses the code that powers the OSV platform.

[Visit the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features:

*   **Comprehensive Vulnerability Data:** Centralized database with detailed vulnerability information.
*   **Open and Accessible:** Free and open-source, fostering community collaboration.
*   **Unified Data Format:** Standardized vulnerability descriptions for easy consumption.
*   **API Access:** Robust API for integrating vulnerability data into your tools.
*   **Web UI:** User-friendly web interface to explore vulnerabilities at [https://osv.dev](https://osv.dev).
*   **Scanner Tool:** A Go-based scanner to detect vulnerabilities in your project dependencies, detailed in its own repository [here](https://github.com/google/osv-scanner).

## Core Components:

This repository contains the following key components:

*   **`deployment/`:** Infrastructure-as-code configuration for GCP deployment.
*   **`docker/`:** Dockerfiles for CI and various services.
*   **`docs/`:** Documentation source code (Jekyll) and tools.
*   **`gcp/api`:** OSV API server implementation (including protobuf files).
*   **`gcp/datastore`:** Datastore index configuration.
*   **`gcp/functions`:** Cloud Functions (e.g., for PyPI vulnerability publishing).
*   **`gcp/indexer`:** Version determination indexer.
*   **`gcp/website`:** Backend for the OSV web UI, with frontend code.
*   **`gcp/workers/`:** Background workers for tasks like bisection and import.
*   **`osv/`:** Core OSV Python library and ecosystem helpers.
*   **`tools/`:** Development scripts and utilities.
*   **`vulnfeeds/`:** Go module for vulnerability data conversion (e.g., NVD).

To build locally, update the submodules:

```bash
git submodule update --init --recursive
```

## Documentation and API

*   Comprehensive documentation is available at [https://google.github.io/osv.dev](https://google.github.io/osv.dev).
*   API documentation can be found at [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/).

## Data Dumps

Access vulnerability data dumps from the GCS bucket: `gs://osv-vulnerabilities`.  Find more information in [the documentation](https://google.github.io/osv.dev/data/#data-dumps).

## Contribute

We welcome contributions!  Learn more about:

*   [Contributing Code](CONTRIBUTING.md#contributing-code)
*   [Contributing Data](CONTRIBUTING.md#contributing-data)
*   [Contributing Documentation](CONTRIBUTING.md#contributing-documentation)

Join the conversation on the [mailing list](https://groups.google.com/g/osv-discuss).

Have questions or suggestions? [Open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

The OSV project has integrations with several community-built tools. Please consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) to determine suitability for your use.  Examples include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)