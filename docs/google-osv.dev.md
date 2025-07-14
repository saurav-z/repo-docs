[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/osv.dev/badge)](https://scorecard.dev/viewer/?uri=github.com/google/osv.dev)

# OSV: Open Source Vulnerability Database

**OSV is an open-source project dedicated to improving the security of the open-source ecosystem by providing a comprehensive and accessible database of vulnerabilities.** This repository houses the code for running [osv.dev](https://osv.dev), a valuable resource for identifying and mitigating security risks in your software dependencies.  Learn more and contribute at the [original repository](https://github.com/google/osv.dev).

## Key Features

*   **Comprehensive Vulnerability Database:** Access a vast collection of known vulnerabilities affecting open-source projects.
*   **Dependency Scanning:** Utilize the OSV scanner to scan your project's dependencies and identify potential vulnerabilities.
*   **Web UI:** Explore the OSV database and its features through the user-friendly web interface at [osv.dev](https://osv.dev).
*   **Data Dumps:** Download data dumps from a GCS bucket to integrate with your security workflows.
*   **API Access:** Integrate OSV data into your tools and workflows via the OSV API, documented [here](https://google.github.io/osv.dev/api/).

## Repository Structure

This repository contains the necessary code and configuration to operate the OSV platform on Google Cloud Platform (GCP). Key directories include:

*   `deployment/`: Terraform and Cloud Deploy configurations.
*   `docker/`: Dockerfiles for CI and deployment.
*   `docs/`: Documentation files, including the Jekyll-based website.
*   `gcp/api`: OSV API server files.
*   `gcp/datastore`: Datastore index configurations.
*   `gcp/website`: Backend for the osv.dev web interface.
*   `osv/`: Core OSV Python library.
*   `vulnfeeds/`: Go module for vulnerability data conversion.

## Getting Started

To begin, clone the repository and initialize submodules:

```bash
git submodule update --init --recursive
```

## Contributing

We welcome contributions!  See our [CONTRIBUTING.md](https://github.com/google/osv.dev/blob/main/CONTRIBUTING.md) for information on how to contribute code, data, and documentation.  Join the community on our [mailing list](https://groups.google.com/g/osv-discuss). Have a question or suggestion?  Please [open an issue](https://github.com/google/osv.dev/issues).

## Third-Party Tools & Integrations

OSV integrates with many third-party tools, allowing users to integrate vulnerability data into their existing workflows.  Note that these tools are community-built and are not supported or endorsed by OSV maintainers.  Consider consulting the [OpenSSF's Concise Guide for Evaluating Open Source Software](https://best.openssf.org/Concise-Guide-for-Evaluating-Open-Source-Software) to determine suitability for your use.  Some popular tools include:

*   Cortex XSOAR
*   dep-scan
*   Dependency-Track
*   GUAC
*   OSS Review Toolkit
*   pip-audit
*   Renovate
*   Trivy

## Documentation

*   Comprehensive documentation is available [here](https://google.github.io/osv.dev).
*   API documentation is available [here](https://google.github.io/osv.dev/api/).
*   Data Dumps are available from a GCS bucket at `gs://osv-vulnerabilities`. More information is available [here](https://google.github.io/osv.dev/data/#data-dumps).