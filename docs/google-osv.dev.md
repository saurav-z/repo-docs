[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV (Open Source Vulnerability) is a comprehensive database for open-source vulnerabilities, helping you identify and mitigate risks in your software supply chain.**  This repository hosts the infrastructure behind the OSV database and platform.

[Visit the original repository on GitHub](https://github.com/google/osv.dev)

## Key Features

*   **Comprehensive Vulnerability Data:** Access a vast database of known vulnerabilities affecting open-source projects.
*   **Dependency Scanning:** Utilize the OSV scanner tool to identify vulnerable dependencies within your projects.
*   **API Access:** Leverage the OSV API to integrate vulnerability data into your security tools and workflows.
*   **Web UI:** Explore the OSV database and search for vulnerabilities through the user-friendly web interface at [https://osv.dev](https://osv.dev).
*   **Data Dumps:** Access data dumps for offline analysis and integration into your own systems.

## Core Components of this Repository

This repository contains the code for running the OSV platform on Google Cloud Platform (GCP).  Key directories include:

*   `deployment/`: Terraform and Cloud Deploy configuration files.
*   `docker/`: Dockerfiles for CI and deployment.
*   `docs/`: Documentation source files (Jekyll).
*   `gcp/api`: OSV API server files.
*   `gcp/datastore`: Datastore index configuration.
*   `gcp/indexer`: Version determination indexer.
*   `gcp/website`: Backend for the OSV web interface.
*   `gcp/workers/`: Background workers for vulnerability processing and analysis.
*   `osv/`: The core OSV Python library.
*   `tools/`: Development and utility scripts.
*   `vulnfeeds/`: Go modules for vulnerability data conversion.

**Note:**  To build locally, you'll need to initialize submodules:

```bash
git submodule update --init --recursive
```

## Documentation & Resources

*   **Comprehensive Documentation:** [https://google.github.io/osv.dev](https://google.github.io/osv.dev)
*   **API Documentation:** [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/)
*   **Data Dumps:** Available from a GCS bucket at `gs://osv-vulnerabilities`.  Learn more in the [documentation](https://google.github.io/osv.dev/data/#data-dumps).
*   **OSV Scanner:**  The scanner tool is available in its [own repository](https://github.com/google/osv-scanner).

## Contributing

We welcome contributions!  Please see the [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

*   [Contributing Code](CONTRIBUTING.md#contributing-code)
*   [Contributing Data](CONTRIBUTING.md#contributing-data)
*   [Contributing Documentation](CONTRIBUTING.md#contributing-documentation)

For questions and discussions, please use the [mailing list](https://groups.google.com/g/osv-discuss) or [open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV integrates with a variety of community-built tools.  Note that these are not officially endorsed by the OSV project.  Consult the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) for suitability.  Some examples include:

*   [Cortex XSOAR](https://github.com/demisto/content)
*   [dep-scan](https://github.com/AppThreat/dep-scan)
*   [Dependency-Track](https://github.com/DependencyTrack/dependency-track)
*   [GUAC](https://github.com/guacsec/guac)
*   [OSS Review Toolkit](https://github.com/oss-review-toolkit/ort)
*   [pip-audit](https://github.com/pypa/pip-audit)
*   [Renovate](https://github.com/renovatebot/renovate)
*   [Trivy](https://github.com/aquasecurity/trivy)