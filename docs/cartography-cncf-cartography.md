# Cartography: Visualize and Analyze Your Infrastructure with a Powerful Graph Database

**Cartography, a CNCF sandbox project, empowers you to understand and secure your cloud and on-premise infrastructure by mapping assets and their relationships in an intuitive graph view.**  [Explore the original repository on GitHub](https://github.com/cartography-cncf/cartography).

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/cartography-cncf/cartography/badge)](https://scorecard.dev/viewer/?uri=github.com/cartography-cncf/cartography)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9637/badge)](https://www.bestpractices.dev/projects/9637)
[![Build Status](https://github.com/cartography-cncf/cartography/actions/workflows/publish-to-ghcr-and-pypi.yml/badge.svg)](https://github.com/cartography-cncf/cartography/actions/workflows/publish-to-ghcr-and-pypi.yml)

![Cartography Visualization](docs/root/images/accountsandrds.png)

## Key Features

*   **Comprehensive Asset Mapping:**  Ingests and visualizes infrastructure assets from various platforms.
*   **Relationship Discovery:**  Reveals hidden dependencies and connections between your assets.
*   **Intuitive Graph View:** Uses a [Neo4j](https://www.neo4j.com) graph database for easy exploration and analysis.
*   **Extensible and Generic:** Adaptable to any platform or technology, providing a flexible solution for security and infrastructure exploration.
*   **Automated Analysis:**  Supports automated workflows through APIs and integrations.

## Why Use Cartography?

Cartography goes beyond basic inventory management, providing in-depth insights for:

*   **Security Validation:** Identify potential attack paths and validate security assumptions.
*   **Risk Assessment:**  Uncover hidden vulnerabilities and understand your overall risk posture.
*   **Automation:** Build custom applications and automate security tasks using the graph data.
*   **Platform Support:** Integrate with a wide range of platforms, including:

## Supported Platforms & Data Sources

*   **Cloud Providers:**
    *   AWS
    *   GCP
    *   Azure
    *   DigitalOcean
    *   Oracle Cloud Infrastructure
    *   Scaleway
*   **Identity & Access Management (IAM):**
    *   Okta
    *   Microsoft Entra ID
    *   Duo
    *   Keycloak
*   **Containerization & Orchestration:**
    *   Kubernetes
*   **Code Repositories:**
    *   GitHub
*   **Other Integrations:**
    *   Anthropic
    *   Airbyte
    *   BigFix
    *   Cloudflare
    *   Crowdstrike Falcon
    *   Google GSuite
    *   Kandji
    *   Lastpass
    *   NIST CVE
    *   PagerDuty
    *   SentinelOne
    *   SnipeIT
    *   Tailscale
    *   Trivy Scanner

## Getting Started

*   **Installation:** Easily install and configure Cartography.
*   **Querying:** Utilize a user-friendly interface and powerful Cypher query language to explore your data.
*   **Documentation:** Detailed documentation available [here](https://cartography-cncf.github.io/cartography/)

## Community & Contributing

*   **Slack:** Join the `#cartography` channel on the CNCF Slack [here](https://communityinviter.com/apps/cloud-native/cncf)
*   **Community Meetings:** Participate in our monthly meetings for updates and discussions (see [meeting info](https://zoom-lfx.platform.linuxfoundation.org/meetings/cartography?view=week)).
*   **Contributions:** We welcome contributions!  See the [developer documentation](https://cartography-cncf.github.io/cartography/dev/developer-guide.html) and [CONTRIBUTING](#contributing) guidelines for more information.
*   **Code of Conduct:**  All contributors adhere to the [CNCF Code of Conduct](https://github.com/cncf/foundation/blob/main/code-of-conduct.md).

##  Who Uses Cartography?

Join the ranks of organizations that leverage Cartography for infrastructure visualization and security:

1.  [Lyft](https://www.lyft.com)
2.  [Thought Machine](https://thoughtmachine.net/)
3.  [MessageBird](https://messagebird.com)
4.  [Cloudanix](https://www.cloudanix.com/)
5.  [Corelight](https://www.corelight.com/)
6.  [SubImage](https://subimage.io)
7.  {Your company here} :-)

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

---