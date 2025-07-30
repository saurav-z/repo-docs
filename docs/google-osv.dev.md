[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV is a comprehensive, open-source database and API for tracking and analyzing vulnerabilities in open-source software.** This project, hosted on GitHub at [https://github.com/google/osv.dev](https://github.com/google/osv.dev), provides valuable resources for developers, security researchers, and anyone interested in securing the software supply chain.

## Key Features

*   **Centralized Vulnerability Database:** OSV aggregates vulnerability data from various sources, offering a single source of truth.
*   **Open API:** Access the OSV database programmatically via a well-defined API, enabling integration with security tools and workflows.
*   **Dependency Scanning:** Identify vulnerable dependencies using the OSV scanner tool, available in its [own repository](https://github.com/google/osv-scanner).
*   **Web UI:** Browse and explore vulnerabilities through the user-friendly web interface at <https://osv.dev>.
*   **Data Dumps:** Access raw vulnerability data dumps for offline analysis and integration with custom tools (available at `gs://osv-vulnerabilities`; see documentation).

## This Repository: Project Structure

This repository contains the code and configurations for running the OSV infrastructure on Google Cloud Platform (GCP). Key directories include:

*   `deployment/`: Terraform and Cloud Deploy configurations for infrastructure deployment.
*   `docker/`: Docker files for CI/CD and worker images.
*   `docs/`:  Jekyll files for the OSV documentation website.
*   `gcp/api`: OSV API server files, including protobuf definitions.
*   `gcp/datastore`: Datastore index configurations.
*   `gcp/website`: Backend code for the OSV web interface.
*   `osv/`: The core OSV Python library.
*   `vulnfeeds/`: Modules and tools for converting vulnerability data from various sources (e.g., NVD, Alpine, Debian).
*   And more (see original README for details).

**Note:** Many local build steps require the use of submodules.  Ensure you initialize and update them using:

```bash
git submodule update --init --recursive
```

## Documentation and Resources

*   **Comprehensive Documentation:**  Learn more about OSV at [https://google.github.io/osv.dev](https://google.github.io/osv.dev).
*   **API Documentation:** Explore the OSV API at [https://google.github.io/osv.dev/api/](https://google.github.io/osv.dev/api/).

## Contributing

We welcome contributions!  Review our guidelines for contributing [code](CONTRIBUTING.md#contributing-code), [data](CONTRIBUTING.md#contributing-data), and [documentation](CONTRIBUTING.md#contributing-documentation).

*   **Mailing List:** Join the discussion at [https://groups.google.com/g/osv-discuss](https://groups.google.com/g/osv-discuss).
*   **Issue Tracker:** Report bugs, suggest features, or ask questions by [opening an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools and Integrations

OSV is used by a wide range of community-developed tools.  Note that these are not officially supported or endorsed by the OSV maintainers.  Refer to the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) for guidance.  Popular third-party integrations include:

*   Cortex XSOAR
*   dep-scan
*   Dependency-Track
*   GUAC
*   OSS Review Toolkit
*   pip-audit
*   Renovate
*   Trivy