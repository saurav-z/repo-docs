<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg">
</picture>

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV (Open Source Vulnerability) is a comprehensive database and API for open-source vulnerability information, providing critical insights into software security.**

This repository powers the OSV database and its associated infrastructure, offering a centralized resource for vulnerability data and tools to help you secure your software supply chain. Explore the OSV project on [GitHub](https://github.com/google/osv.dev).

## Key Features

*   **Comprehensive Vulnerability Data:** Access a vast database of known vulnerabilities affecting open-source software.
*   **OSV API:** Leverage a powerful API for programmatic access to vulnerability information, enabling integration with security tools and workflows.
*   **OSV Scanner:** Identify vulnerabilities in your project dependencies using the OSV scanner tool (available in a [separate repository](https://github.com/google/osv-scanner)).
*   **Data Dumps:** Download data dumps from a Google Cloud Storage (GCS) bucket for offline analysis and integration.
*   **Web UI:** Browse and search vulnerabilities through the user-friendly OSV web interface at <https://osv.dev>.

## Resources

*   **Documentation:** Detailed documentation is available to guide you through OSV's features and usage.
    *   [Comprehensive Documentation](https://google.github.io/osv.dev)
    *   [API Documentation](https://google.github.io/osv.dev/api/)
    *   [Data Dumps](https://google.github.io/osv.dev/data/#data-dumps)

## Repository Structure

This repository contains the code and configuration for the OSV infrastructure, including:

*   **`bindings/`**: Language bindings for the OSV API (currently Go only).
*   **`deployment/`**: Infrastructure-as-code configuration for deployment.
*   **`docker/`**: Dockerfiles for CI and deployment.
*   **`docs/`**: Documentation files.
*   **`gcp/api`**: OSV API server files.
*   **`gcp/datastore`**: Datastore index configuration.
*   **`gcp/functions`**: Cloud Functions for vulnerability processing.
*   **`gcp/indexer`**: Version determination indexer.
*   **`gcp/website`**: Backend for the OSV web interface.
*   **`gcp/workers/`**: Background worker processes.
*   **`osv/`**: Core OSV Python library.
*   **`tools/`**: Development and utility scripts.
*   **`vulnfeeds/`**: Vulnerability feed converters (e.g., NVD, Alpine, Debian).

To build and run locally, update submodules:

```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions! Learn more about contributing to code, data, and documentation in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

*   **[Contributing Guide](CONTRIBUTING.md)**
*   **Mailing List:** [OSV Discussion](https://groups.google.com/g/osv-discuss)
*   **Issues:** [Open an Issue](https://github.com/google/osv.dev/issues)

## Third-Party Tools & Integrations

OSV integrates with a variety of third-party tools, providing added functionality and flexibility. Note that these are community-built and not officially supported.

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)