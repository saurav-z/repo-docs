[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database and Infrastructure

**OSV.dev provides a comprehensive, centralized database of open-source vulnerabilities and a suite of tools to help you identify and mitigate risks.**

This repository contains the code for the OSV infrastructure, powering the [OSV.dev website](https://osv.dev) and related services.

[Visit the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Centralized Vulnerability Database:** Access a comprehensive and curated database of open-source vulnerabilities.
*   **Vulnerability Scanning:** Integrates with the [OSV scanner](https://github.com/google/osv-scanner) to scan your dependencies for known vulnerabilities.
*   **Web UI:** Explore vulnerabilities and related information through the user-friendly web interface at [OSV.dev](https://osv.dev).
*   **API Access:** Utilize the OSV API to programmatically access vulnerability data and integrate it into your security workflows ([API Documentation](https://google.github.io/osv.dev/api/)).
*   **Data Dumps:** Access data dumps from a GCS bucket `gs://osv-vulnerabilities` ([Data Dump Documentation](https://google.github.io/osv.dev/data/#data-dumps)).

## Repository Structure

This repository is organized as follows:

*   `deployment/`: Terraform & Cloud Deploy configuration files and Cloud Build config yamls.
*   `docker/`: CI docker files and worker base images.
*   `docs/`: Jekyll files for the OSV documentation.
*   `gcp/api`: OSV API server files and protobuf definitions.
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/functions`: Cloud Function for publishing PyPI vulnerabilities.
*   `gcp/indexer`: Version indexing logic.
*   `gcp/website`: Backend of the OSV web interface.
*   `gcp/workers/`: Workers for various background tasks (bisection, import, export, alias) and cron jobs.
*   `osv/`: The core OSV Python library.
*   `tools/`: Utility scripts for development and maintenance.
*   `vulnfeeds/`: Go module for vulnerability data conversion (NVD, Alpine, Debian).

### Submodules

To build locally, update the submodules:

```bash
git submodule update --init --recursive
```

## Contributing

Contributions are welcome! Please see the following resources:

*   [Contributing Code](CONTRIBUTING.md#contributing-code)
*   [Contributing Data](CONTRIBUTING.md#contributing-data)
*   [Contributing Documentation](CONTRIBUTING.md#contributing-documentation)

Join the discussion on the [mailing list](https://groups.google.com/g/osv-discuss).

## Questions and Suggestions

Open an issue on [GitHub](https://github.com/google/osv.dev/issues).

## Third-Party Tools & Integrations

Explore community-built tools that integrate with OSV:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)

*Note: These are community-built tools and are not supported or endorsed by OSV maintainers. Evaluate their suitability using the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software).*