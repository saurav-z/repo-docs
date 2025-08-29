<picture>
    <source srcset="docs/images/osv_logo_dark-full.svg"  media="(prefers-color-scheme: dark)">
    <!-- markdown-link-check-disable-next-line -->
    <img src="docs/images/osv_logo_light-full.svg" alt="OSV Logo">
</picture>

---

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

## OSV: Open Source Vulnerability Database

**OSV (Open Source Vulnerability) is a comprehensive, open-source database for tracking and analyzing vulnerabilities in open-source software, offering a unified and accessible source of vulnerability information.** [Learn more and explore the OSV project on GitHub](https://github.com/google/osv.dev).

**Key Features of OSV:**

*   **Comprehensive Vulnerability Data:** Access a vast database of known vulnerabilities across numerous open-source projects.
*   **API Access:**  Integrate vulnerability data directly into your security tools and workflows using the OSV API ([API Documentation](https://google.github.io/osv.dev/api/)).
*   **Dependency Scanning with OSV Scanner:**  Identify vulnerabilities in your project's dependencies using the [OSV Scanner](https://github.com/google/osv-scanner).
*   **Data Dumps:** Download data dumps for offline analysis and integration ([Data Dump Documentation](https://google.github.io/osv.dev/data/#data-dumps)).
*   **Web UI:** Browse and search vulnerabilities through an intuitive web interface at <https://osv.dev>.

## Project Structure

This repository houses the infrastructure and code that powers the OSV platform. Key directories include:

*   `deployment/`:  Deployment configurations (Terraform, Cloud Deploy, Cloud Build).
*   `docker/`: Dockerfiles for CI, deployment, and worker images.
*   `docs/`:  Documentation (Jekyll files).
*   `gcp/api`: OSV API server code (Go, Protobuf).
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/website`: Backend for the OSV web interface.
*   `osv/`: Core OSV Python library and related components.
*   `vulnfeeds/`: Go modules for vulnerability data conversion.

**To build locally, you'll need to initialize submodules:**

```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions!  Learn how to contribute to [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation).

*   **Questions and Suggestions:**  [Open an issue](https://github.com/google/osv.dev/issues).
*   **Mailing List:**  [osv-discuss](https://groups.google.com/g/osv-discuss).

## Third-Party Tools and Integrations

OSV integrates with a variety of third-party tools. Note that these community-built tools are not supported or endorsed by OSV maintainers:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)