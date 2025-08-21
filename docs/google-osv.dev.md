[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV.dev: Open Source Vulnerability Database

**OSV.dev is a comprehensive, open-source vulnerability database designed to help you identify and manage security vulnerabilities in your open-source dependencies.**  Visit the original repository on GitHub: [https://github.com/google/osv.dev](https://github.com/google/osv.dev)

## Key Features:

*   **Comprehensive Vulnerability Database:** Access a vast and continuously updated database of known vulnerabilities in open-source software.
*   **Dependency Scanning Tool:** Use the provided Go-based scanner to identify vulnerable dependencies within your projects.
*   **Web UI:**  Explore the vulnerability database and related information via the user-friendly web interface at [https://osv.dev](https://osv.dev).
*   **API Access:**  Integrate OSV data directly into your security workflows using the robust API.  Find API documentation [here](https://google.github.io/osv.dev/api/).
*   **Data Dumps:** Access data dumps from a GCS bucket (`gs://osv-vulnerabilities`) for offline analysis and integration.  See [documentation](https://google.github.io/osv.dev/data/#data-dumps) for details.
*   **Community Driven:** Benefit from a collaborative ecosystem with contributions welcome.

## This Repository's Contents

This repository contains the infrastructure and code that powers the OSV.dev platform on Google Cloud Platform (GCP). Key directories include:

*   `deployment/`: Terraform and Cloud Deploy configuration.
*   `docker/`: Docker files for CI and deployment.
*   `docs/`:  Documentation files (Jekyll).
*   `gcp/api`: OSV API server files.
*   `gcp/datastore`: Datastore index file.
*   `gcp/functions`: Cloud Functions for PyPI vulnerability publishing.
*   `gcp/indexer`: Version determination indexer.
*   `gcp/website`: Backend for the web interface.
*   `gcp/workers/`: Background workers for vulnerability analysis.
*   `osv/`: Core OSV Python library.
*   `tools/`: Development scripts and tools.
*   `vulnfeeds/`: Go module for vulnerability feed conversion.

**Note:**  You'll need to initialize submodules for local building to work correctly: `git submodule update --init --recursive`

## Contributing

We welcome contributions from the community! Learn more about contributing:

*   [Code](CONTRIBUTING.md#contributing-code)
*   [Data](CONTRIBUTING.md#contributing-data)
*   [Documentation](CONTRIBUTING.md#contributing-documentation)

Join the conversation on the [mailing list](https://groups.google.com/g/osv-discuss).  For questions or suggestions, [open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools & Integrations

OSV.dev is compatible with many third-party tools and integrations, including:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

*Note:* These tools are community-built and not officially endorsed by OSV.dev maintainers.  Consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) for suitability.