[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: The Open Source Vulnerability Database

**OSV is a free and open-source vulnerability database, API, and associated tools to help you understand and mitigate security risks in your open-source dependencies.** Developed by Google, OSV provides a centralized, comprehensive, and reliable source of vulnerability information. Check out the [original repo here](https://github.com/google/osv.dev).

## Key Features

*   **Comprehensive Vulnerability Data:** Access a constantly updated database of known vulnerabilities across various open-source projects.
*   **API Access:** Integrate OSV data seamlessly into your security tools and workflows via a robust API.
*   **Dependency Scanning Tool:** Utilize a Go-based scanner to identify vulnerabilities in your project's dependencies by comparing them against the OSV database. The scanner can scan lockfiles, SBOMs, and git repositories. See the scanner [here](https://github.com/google/osv-scanner).
*   **Web UI:** Browse and explore the OSV vulnerability database through a user-friendly web interface at [https://osv.dev](https://osv.dev).
*   **Data Dumps:** Access data dumps from a GCS bucket for offline analysis and integration into your custom tools.

## Documentation & Resources

*   **Comprehensive Documentation:** [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
*   **API Documentation:** [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)
*   **Data Dumps:** Access the data dumps available at `gs://osv-vulnerabilities`. For more information, check out the [documentation](https://google.github.io/osv.dev/data/#data-dumps).

## Repository Structure

This repository contains the code and configuration for running the OSV platform on Google Cloud Platform (GCP). Key components include:

*   `deployment/`: Terraform and Cloud Deploy configuration.
*   `docker/`: Dockerfiles for CI, deployment, and worker images.
*   `docs/`: Documentation (Jekyll) and build scripts.
*   `gcp/api`: OSV API server code.
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Functions for vulnerability publishing.
*   `gcp/indexer`: Version determination indexer.
*   `gcp/website`: Backend for the osv.dev web interface.
*   `gcp/workers/`: Workers for various tasks, including bisection and impact analysis.
*   `osv/`: Core OSV Python library.
*   `tools/`: Development scripts and tools.
*   `vulnfeeds/`: Modules for vulnerability feed conversions.

**Note:** You may need to initialize submodules using `git submodule update --init --recursive` for local building.

## Contributing

We welcome contributions!  Learn more about contributing [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation). Discuss ideas on the [mailing list](https://groups.google.com/g/osv-discuss).  For questions and suggestions, please [open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV integrates with a variety of third-party tools. Please note that these are community-built tools and are not supported or endorsed by OSV maintainers.  Consider the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) when evaluating these tools.

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)