[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV.dev: Open Source Vulnerability Database

**OSV.dev is a comprehensive, free, and open-source database of vulnerabilities affecting open-source software, helping developers identify and address security risks in their projects.**

[Visit the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Centralized Vulnerability Database:** A single source of truth for known vulnerabilities in open-source packages.
*   **Open and Free:**  Available for anyone to use and contribute to.
*   **Comprehensive Coverage:**  Supports a wide range of ecosystems and package managers.
*   **API Access:**  Programmatic access to vulnerability data via a REST API.
*   **Web UI:** User-friendly web interface for searching and exploring vulnerabilities (<https://osv.dev>).
*   **Scanner Tool:** Go-based scanner tool to identify vulnerabilities in your project dependencies (located in [osv-scanner](https://github.com/google/osv-scanner)).
*   **Data Dumps:** Access data dumps for offline analysis and integration.

## Documentation & Resources

*   **Comprehensive Documentation:** [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
*   **API Documentation:** [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)
*   **Data Dumps:** Available at `gs://osv-vulnerabilities`.  See the [documentation](https://google.github.io/osv.dev/data/#data-dumps) for details.

## Repository Structure

This repository contains the code for running the OSV.dev platform on Google Cloud Platform (GCP). Key directories include:

*   `deployment/`: Terraform & Cloud Deploy configuration.
*   `docker/`: Docker files for CI and deployment.
*   `docs/`: Documentation files (Jekyll).
*   `gcp/api`: OSV API server code.
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Functions (e.g., PyPI vulnerability publishing).
*   `gcp/indexer`: Version determination indexer.
*   `gcp/website`: Backend of the web interface.
*   `gcp/workers/`: Background workers (bisection, import, export, alias).
*   `osv/`: Core OSV Python library.
*   `tools/`: Development scripts and tools.
*   `vulnfeeds/`:  NVD CVE conversion module.

**Note:** You will need to initialize submodules to build locally: `git submodule update --init --recursive`

## Contributing

We welcome contributions!  Learn more about how to contribute:

*   **Code:** [CONTRIBUTING.md#contributing-code](CONTRIBUTING.md#contributing-code)
*   **Data:** [CONTRIBUTING.md#contributing-data](CONTRIBUTING.md#contributing-data)
*   **Documentation:** [CONTRIBUTING.md#contributing-documentation](CONTRIBUTING.md#contributing-documentation)
*   **Mailing List:** [https://groups.google.com/g/osv-discuss](https://groups.google.com/g/osv-discuss)
*   **Open an issue:** [https://github.com/google/osv.dev/issues](https://github.com/google/osv.dev/issues)

## Third-Party Tools & Integrations

OSV integrates with a variety of third-party tools.  These are community-built and not officially supported by OSV.  See the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) for evaluating suitability.  Examples include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)