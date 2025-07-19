[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV is an open-source vulnerability database, helping you identify and address vulnerabilities in your open-source dependencies.**

[View the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Comprehensive Vulnerability Data:** Access a curated database of known vulnerabilities impacting open-source software.
*   **Dependency Scanning:** Utilize the OSV scanner to identify vulnerable dependencies in your projects.
*   **API Access:** Integrate OSV data into your security workflows via the OSV API.
*   **Web UI:** Explore vulnerabilities and project information through the OSV web interface.
*   **Data Dumps:** Download OSV data for offline analysis and integration.

## Resources

*   **Documentation:** [Comprehensive documentation](https://google.github.io/osv.dev)
*   **API Documentation:** [API documentation](https://google.github.io/osv.dev/api/)
*   **Web UI:** [OSV Web UI](https://osv.dev)
*   **Data Dumps:** [Data dumps](https://google.github.io/osv.dev/data/#data-dumps)
*   **OSV Scanner:** The OSV scanner is available in its [own repository](https://github.com/google/osv-scanner).

## Repository Structure

This repository houses the core code and infrastructure for running the OSV project. Key components include:

*   **Deployment:** Configuration files for deployment on Google Cloud Platform (GCP).
*   **Docker:** Dockerfiles for building various images, including CI and worker base images.
*   **Docs:** Files for building and maintaining the project documentation.
*   **GCP:** Code related to the OSV API server, datastore, Cloud Functions, and workers.
*   **OSV:** The core OSV Python library.
*   **Vulnfeeds:** Modules for converting vulnerability feeds from various sources.

To build locally, you may need to initialize submodules:

```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions! Learn more about:

*   [Contributing Code](CONTRIBUTING.md#contributing-code)
*   [Contributing Data](CONTRIBUTING.md#contributing-data)
*   [Contributing Documentation](CONTRIBUTING.md#contributing-documentation)

Join the discussion on the [mailing list](https://groups.google.com/g/osv-discuss) and report issues via the [issue tracker](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

The OSV project is supported by a vibrant community. Note that these are community-built tools and are not supported or endorsed by the core OSV maintainers. Explore the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) to determine suitability for your use. Popular tools include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)