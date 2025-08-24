[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database and Infrastructure

**OSV (Open Source Vulnerability) is a collaborative, open-source project providing a comprehensive database and infrastructure for tracking and mitigating vulnerabilities in open-source software.**

[Explore the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Comprehensive Vulnerability Database:** Access a curated database of known vulnerabilities affecting open-source projects.
*   **Vulnerability Scanning:** Utilize the OSV scanner to identify vulnerabilities in your project dependencies.
*   **API Access:** Integrate OSV data into your security tools and workflows through a robust API.
*   **Web UI:** Easily search and explore the OSV database through a user-friendly web interface.
*   **Data Dumps:** Download regularly updated data dumps for offline analysis and integration.
*   **Community Driven:** Benefit from a community-driven project with open contributions and a dedicated mailing list.

## Documentation

*   Comprehensive documentation is available at: [https://google.github.io/osv.dev](https://google.github.io/osv.dev).
*   API documentation is available at: [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/).

## Data Dumps

*   Data dumps are available from a GCS bucket at `gs://osv-vulnerabilities`.
*   For more information, check out the documentation: [https://google.github.io/osv.dev/data/#data-dumps](https://google.github.io/osv.dev/data/#data-dumps).

## Using the Web UI

*   An instance of the OSV web UI is deployed at <https://osv.dev>.

## Using the Scanner

*   The Go-based OSV scanner is available in its [own repository](https://github.com/google/osv-scanner).
*   It can scan various lockfiles, Debian Docker containers, SPDX and CycloneDB SBOMs, and Git repositories.

## Repository Structure

This repository houses the code for running the OSV infrastructure on Google Cloud Platform (GCP), including:

*   `deployment/`: Terraform & Cloud Deploy configuration
*   `docker/`: CI docker files and base images
*   `docs/`: Jekyll files for the OSV documentation
*   `gcp/api`: OSV API server files
*   `gcp/datastore`: Datastore index configuration
*   `gcp/functions`: Cloud Function for publishing PyPI vulnerabilities
*   `gcp/indexer`: Version determination indexer
*   `gcp/website`: Backend of the osv.dev web interface
*   `gcp/workers/`: Workers for bisection and impact analysis
*   `osv/`: Core OSV Python library
*   `tools/`: Development scripts and tools
*   `vulnfeeds/`: Go module for NVD CVE conversion

**To build locally, be sure to update submodules:**

```bash
git submodule update --init --recursive
```

## Contributing

Contributions are welcome! Learn more about contributing [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation).

*   Join the discussion: [mailing list](https://groups.google.com/g/osv-discuss).
*   Have a question or suggestion? Please [open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

The OSV project is supported by a vibrant community, with many third-party tools integrating with its data. Note that these tools are community-built and not officially endorsed by OSV maintainers. Consider the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) for your security assessments. Some popular tools include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)