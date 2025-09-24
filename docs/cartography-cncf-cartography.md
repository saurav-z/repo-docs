# Cartography: Visualize and Understand Your Cloud Infrastructure

**Cartography is a powerful, open-source tool that maps your cloud infrastructure and relationships, revealing hidden security risks through an intuitive graph view.** Discover valuable insights into your cloud environment and streamline security efforts with Cartography - [View the original repo on GitHub](https://github.com/cartography-cncf/cartography).

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/cartography-cncf/cartography/badge)](https://scorecard.dev/viewer/?uri=github.com/cartography-cncf/cartography)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9637/badge)](https://www.bestpractices.dev/projects/9637)
![build](https://github.com/cartography-cncf/cartography/actions/workflows/publish-to-ghcr-and-pypi.yml/badge.svg)
![Cartography](docs/root/images/logo-horizontal.png)

![Visualization of RDS nodes and AWS nodes](docs/root/images/accountsandrds.png)

## Key Features

*   **Graph-Based Visualization:** Explore your infrastructure assets and their connections using an interactive graph database (powered by Neo4j).
*   **Multi-Platform Support:** Ingests data from a wide variety of cloud providers and services.
*   **Dependency Mapping:** Uncover hidden relationships between your assets to identify potential attack vectors.
*   **Extensible Architecture:** Customize and extend Cartography with your own plugins to fit your specific needs.
*   **Automated Analysis:** Integrate Cartography into your automation pipelines for continuous security monitoring and analysis.
*   **Web Frontend and API:** Access and interact with your infrastructure data through a user-friendly web interface or via API calls.

## Why Use Cartography?

Cartography empowers you to:

*   **Enhance Security:** Identify and mitigate security risks by visualizing potential attack paths.
*   **Improve Compliance:** Generate asset reports to validate security assumptions and demonstrate compliance.
*   **Accelerate Incident Response:** Quickly understand your infrastructure during incidents by exploring the relationships between assets.
*   **Gain Deep Insights:** Discover cross-tenant permissions and network paths within your environment.

## Supported Platforms

Cartography supports a growing list of platforms, including:

*   Airbyte
*   Amazon Web Services (AWS)
*   Anthropic
*   BigFix
*   Cloudflare
*   Crowdstrike Falcon
*   DigitalOcean
*   Duo
*   GitHub
*   Google Cloud Platform (GCP)
*   Google GSuite
*   Kandji
*   Keycloak
*   Kubernetes
*   Lastpass
*   Microsoft Azure
*   Microsoft Entra ID
*   NIST CVE
*   Okta
*   OpenAI
*   Oracle Cloud Infrastructure (OCI)
*   PagerDuty
*   Scaleway
*   SentinelOne
*   SnipeIT
*   Tailscale
*   Trivy Scanner

*(Click each platform link to view the supported features)*

## Getting Started

### Installation

[Follow these steps](https://cartography-cncf.github.io/cartography/install.html) to quickly set up a test graph and populate it with data.

### Production Setup

For production environments, review [these recommendations](https://cartography-cncf.github.io/cartography/ops.html) on setting up Cartography.

## Usage

### Querying

Use the [querying tutorial](https://cartography-cncf.github.io/cartography/usage/tutorial.html) to get started with querying your graph database. Use the [data schema](https://cartography-cncf.github.io/cartography/usage/schema.html) as a helpful reference.

### Building Applications

Cartography's data can be used to build custom applications and data pipelines. View the doc on [applications](https://cartography-cncf.github.io/cartography/usage/applications.html).

## Docs

Find all the documentation on [Cartography's website](https://cartography-cncf.github.io/cartography/).

## Community

*   **Slack:** Join the CNCF Slack workspace [here](https://communityinviter.com/apps/cloud-native/cncf), and then join the `#cartography` channel.
*   **Community Meetings:** Attend our [monthly community meeting](https://zoom-lfx.platform.linuxfoundation.org/meetings/cartography?view=week).
    *   Meeting minutes are [here](https://docs.google.com/document/d/1VyRKmB0dpX185I15BmNJZpfAJ_Ooobwz0U1WIhjDxvw).
    *   Recorded videos from before 2025 are posted [here](https://www.youtube.com/playlist?list=PLMga2YJvAGzidUWJB_fnG7EHI4wsDDsE1).

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Contributing

We welcome contributions to Cartography!

### Code of Conduct

All contributors and participants must adhere to the [CNCF code of conduct](https://github.com/cncf/foundation/blob/main/code-of-conduct.md).

### Reporting Issues and Suggestions

Submit your [GitHub issue](https://github.com/cartography-cncf/cartography/issues) to report a bug or request a feature.

### Developing Cartography

Get started with our [developer documentation](https://cartography-cncf.github.io/cartography/dev/developer-guide.html). Please feel free to submit your own PRs to update documentation.

## Who Uses Cartography?

1.  Lyft
2.  Thought Machine
3.  MessageBird
4.  Cloudanix
5.  Corelight
6.  SubImage
7.  {Your company here} :-)

*(If your company uses Cartography, please submit a pull request to add your name!)*

---

Cartography is a [Cloud Native Computing Foundation](https://www.cncf.io/) sandbox project.<br>
<div style="background-color: white; display: inline-block; padding: 10px;">
  <img src="docs/root/images/cncf-color.png" alt="CNCF Logo" width="200">
</div>