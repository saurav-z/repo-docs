<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg">
</picture>

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV (Open Source Vulnerability) is a comprehensive, open-source database and API for vulnerability information, designed to help developers identify and mitigate security risks in their open-source dependencies.**  Learn more and contribute at the [OSV.dev repository](https://github.com/google/osv.dev).

## Key Features

*   **Comprehensive Vulnerability Data:** Access a vast, curated database of known vulnerabilities affecting open-source projects.
*   **API Access:** Integrate vulnerability data directly into your tools and workflows using the OSV API. ([API Documentation](https://google.github.io/osv.dev/api/))
*   **Dependency Scanning Tool:** Use the Go-based scanner to identify vulnerable dependencies in your projects. (See [OSV Scanner](https://github.com/google/osv-scanner).)
*   **Data Dumps:** Access data dumps for offline analysis and integration. ([Data Dump Documentation](https://google.github.io/osv.dev/data/#data-dumps))
*   **Web UI:** Explore the OSV vulnerability database through a user-friendly web interface. ([OSV Web UI](https://osv.dev))

## Project Structure

This repository contains the core infrastructure for the OSV project, including:

*   **Deployment configuration:** Terraform and Cloud Deploy configuration files, Cloud Build configs (`deployment/`)
*   **Dockerfiles:** CI, deployment, Terraform, and worker base images (`docker/`)
*   **Documentation:** Jekyll files for the OSV documentation website (`docs/`)
*   **API Server:** OSV API server files, including protobuf definitions (`gcp/api`)
*   **Data Storage:** Datastore index file (`gcp/datastore`)
*   **Cloud Functions:** For publishing vulnerabilities from PyPI (`gcp/functions`)
*   **Indexing and Processing:** The version determination indexer (`gcp/indexer`)
*   **Web Interface Backend:** Backend code for the OSV web interface and blog posts (`gcp/website`)
*   **Background Workers:** For bisection, impact analysis, importing, exporting and alias workers (`gcp/workers/`)
*   **Core Library:** The OSV Python library and related ecosystem tools (`osv/`)
*   **Development Tools:** Scripts and tools for development and maintenance (`tools/`)
*   **Vulnerability Feed Conversion:** Modules for converting vulnerability feeds (`vulnfeeds/`)

To build many local components, ensure you update submodules:

```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions!  Learn how to contribute to [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation).  Join the discussion on the [mailing list](https://groups.google.com/g/osv-discuss).

**Questions or Suggestions?**  Please [open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV integrates with various third-party tools to enhance vulnerability management workflows. These community-built tools are not officially supported or endorsed by OSV maintainers. We recommend consulting the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) for suitability. Some popular integrations include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)