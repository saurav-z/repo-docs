![Cartography](docs/root/images/logo-horizontal.png)

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/cartography-cncf/cartography/badge)](https://scorecard.dev/viewer/?uri=github.com/cartography-cncf/cartography)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9637/badge)](https://www.bestpractices.dev/projects/9637)
![build](https://github.com/cartography-cncf/cartography/actions/workflows/publish-to-ghcr-and-pypi.yml/badge.svg)

## Cartography: Visualize and Understand Your Infrastructure with a Powerful Graph Database

Cartography is a versatile, open-source security tool that maps your entire infrastructure and the relationships between its components using a graph database, providing valuable insights for security, compliance, and automation. [Explore the Cartography repository](https://github.com/cartography-cncf/cartography).

**Key Features:**

*   **Comprehensive Asset Mapping:** Discovers and visualizes infrastructure assets and their connections from various platforms.
*   **Intuitive Graph Visualization:** Presents your infrastructure data in an easy-to-understand graph view, powered by Neo4j.
*   **Extensive Platform Support:** Integrates with a wide range of platforms, including AWS, GCP, Azure, Kubernetes, and many more (see below).
*   **Security-Focused Analysis:** Enables the identification of potential attack paths and security vulnerabilities.
*   **Automated Reporting and APIs:** Allows for automated asset reporting and integration with other tools.
*   **Extensible Architecture:** Designed to be extended with custom plugins for your specific needs.

## Why Use Cartography?

Cartography excels at revealing hidden dependencies and security risks within your infrastructure, allowing you to:

*   **Improve Security Posture:** Identify vulnerabilities and potential attack vectors.
*   **Streamline Compliance:** Generate reports and demonstrate compliance with security standards.
*   **Enhance Automation:** Automate tasks such as asset inventory and vulnerability assessment.
*   **Facilitate Exploration:** Easily explore your infrastructure and uncover unexpected relationships.

## Supported Platforms

Cartography supports a wide range of platforms and services.  Below are examples of supported platforms:

*   **Cloud Providers:** AWS, Azure, GCP, DigitalOcean, Oracle Cloud Infrastructure, Scaleway
*   **Identity & Access Management:**  Microsoft Entra ID, Okta, Duo, Keycloak,  GSuite, Tailscale, Anthropic, OpenAI,
*   **Containerization & Orchestration:** Kubernetes
*   **Code Repositories & CI/CD:** GitHub,
*   **Vulnerability Scanning:** Trivy Scanner, Crowdstrike Falcon
*   **Endpoint and Device Management:** BigFix, Kandji,
*   **Other Services:** Airbyte, Cloudflare, Duo, Lastpass, PagerDuty, SentinelOne, SnipeIT, NIST CVE

*(See the original [README](https://github.com/cartography-cncf/cartography) for a full and detailed list of supported integrations.)*

## Getting Started

*   **Installation:** Get started quickly by following the [installation guide](https://cartography-cncf.github.io/cartography/install.html).
*   **Production Setup:** Learn about deploying Cartography in production in the [operations documentation](https://cartography-cncf.github.io/cartography/ops.html).
*   **Querying & Usage:** Explore your data and leverage Cartography's capabilities through the [querying tutorial](https://cartography-cncf.github.io/cartography/usage/tutorial.html) and the [applications documentation](https://cartography-cncf.github.io/cartography/usage/applications.html).

## Community & Resources

*   **Documentation:** Access comprehensive documentation at [Cartography Documentation](https://cartography-cncf.github.io/cartography/).
*   **Slack:** Join the `#cartography` channel in the CNCF Slack workspace. [Join the CNCF Slack workspace](https://communityinviter.com/apps/cloud-native/cncf).
*   **Community Meetings:** Participate in monthly community meetings to discuss development and usage. [Monthly community meeting details](https://zoom-lfx.platform.linuxfoundation.org/meetings/cartography?view=week)
*   **YouTube:** Watch recordings of past community meetings [here](https://www.youtube.com/playlist?list=PLMga2YJvAGzidUWJB_fnG7EHI4wsDDsE1).

## Contributing

We welcome contributions to Cartography!

*   **Code of Conduct:** All contributors are expected to follow the [CNCF code of conduct](https://github.com/cncf/foundation/blob/main/code-of-conduct.md).
*   **Issues:** Report bugs, request features, or start discussions by submitting a GitHub issue.
*   **Development:** Find detailed information about developing Cartography in the [developer documentation](https://cartography-cncf.github.io/cartography/dev/developer-guide.html).

## Who Uses Cartography?

*   Lyft
*   Thought Machine
*   MessageBird
*   Cloudanix
*   Corelight
*   SubImage
*   {Your company here} :-)

*(If your organization uses Cartography, please file a PR and update this list.)*

---

Cartography is a [Cloud Native Computing Foundation](https://www.cncf.io/) sandbox project.<br>
<div style="background-color: white; display: inline-block; padding: 10px;">
  <img src="docs/root/images/cncf-color.png" alt="CNCF Logo" width="200">
</div>