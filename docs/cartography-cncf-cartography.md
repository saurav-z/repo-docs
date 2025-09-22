# Cartography: Visualize and Analyze Your Infrastructure with a Powerful Graph Database

**Cartography empowers you to understand your infrastructure's relationships and identify potential security risks by visualizing your assets in a Neo4j graph database.**  ([See the original repo](https://github.com/cartography-cncf/cartography))

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/cartography-cncf/cartography/badge)](https://scorecard.dev/viewer/?uri=github.com/cartography-cncf/cartography)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9637/badge)](https://www.bestpractices.dev/projects/9637)
![build](https://github.com/cartography-cncf/cartography/actions/workflows/publish-to-ghcr-and-pypi.yml/badge.svg)

## Key Features

*   **Consolidated View:** Ingests data from various platforms and services to provide a unified view of your infrastructure.
*   **Intuitive Graph Database:**  Leverages Neo4j to visualize relationships between assets, making complex dependencies easy to understand.
*   **Extensible and Customizable:**  Easily extendable with plugins for new data sources and analysis capabilities.
*   **Flexible Exploration:** Offers a web frontend and APIs for both manual exploration and automated security analysis.
*   **Security-Focused:** Helps identify potential attack paths and validate security assumptions.
*   **Platform Support:** Integrates with a wide range of platforms and services.

## Supported Platforms

Cartography supports data ingestion from a comprehensive list of platforms, including:

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

## Why Use Cartography?

Cartography goes beyond simple asset inventories, providing a dynamic and interconnected view of your infrastructure.  Use Cartography to:

*   **Identify Hidden Dependencies:** Uncover relationships between your assets, revealing potential security vulnerabilities.
*   **Improve Security Posture:** Proactively assess and strengthen your security posture by visualizing attack paths.
*   **Streamline Security Operations:** Automate tasks like asset reporting and vulnerability analysis.
*   **Gain Deep Insights:** Ask complex questions about your environment, such as data access control and network paths.
*   **Facilitate Collaboration:** Enable security teams, service owners, and red/blue teams to collaborate more effectively.

## Getting Started

### Installation and Configuration

1.  **Try it out on a test machine:** Follow the instructions [here](https://cartography-cncf.github.io/cartography/install.html) to set up a test graph.
2.  **Set up in production:** Review recommendations [here](https://cartography-cncf.github.io/cartography/ops.html) for production environments.

### Usage

*   **Querying the Database:** Explore your data using the Neo4j graph database query language. Start with the [querying tutorial](https://cartography-cncf.github.io/cartography/usage/tutorial.html). Refer to the [data schema](https://cartography-cncf.github.io/cartography/usage/schema.html) for reference.
*   **Building Applications:** Integrate Cartography into your existing workflows and build custom applications. See the documentation on [applications](https://cartography-cncf.github.io/cartography/usage/applications.html).

## Documentation

Comprehensive documentation is available [here](https://cartography-cncf.github.io/cartography/).

## Community

*   **Slack:** Join the `#cartography` channel in the CNCF Slack workspace ([CNCF Slack invite](https://communityinviter.com/apps/cloud-native/cncf)).
*   **Community Meetings:** Attend the [monthly community meeting](https://zoom-lfx.platform.linuxfoundation.org/meetings/cartography?view=week) to discuss the project. Meeting minutes are [here](https://docs.google.com/document/d/1VyRKmB0dpX185I15BmNJZpfAJ_Ooobwz0U1WIhjDxvw).  Past meeting videos are [here](https://www.youtube.com/playlist?list=PLMga2YJvAGzidUWJB_fnG7EHI4wsDDsE1).

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Contributing

Contributions are welcome!

*   **Code of Conduct:** All contributors must adhere to the [CNCF code of conduct](https://github.com/cncf/foundation/blob/main/code-of-conduct.md).
*   **Bug Reports & Feature Requests:** Submit issues on [GitHub](https://github.com/cartography-cncf/cartography/issues).  Discussions will be moved to [GitHub Discussions](https://github.com/cartography-cncf/cartography/discussions) for broader conversations.
*   **Development:** Refer to the [developer documentation](https://cartography-cncf.github.io/cartography/dev/developer-guide.html) to get started.

## Who Uses Cartography?

*   Lyft
*   Thought Machine
*   MessageBird
*   Cloudanix
*   Corelight
*   SubImage
*   {Your company here} :-)

  Please add your company to this list via a PR and say hi on Slack!

---

Cartography is a [Cloud Native Computing Foundation](https://www.cncf.io/) sandbox project.<br>
<div style="background-color: white; display: inline-block; padding: 10px;">
  <img src="docs/root/images/cncf-color.png" alt="CNCF Logo" width="200">
</div>