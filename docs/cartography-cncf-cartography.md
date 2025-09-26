# Cartography: Visualize and Understand Your Infrastructure with a Graph Database

**Cartography is a powerful open-source tool that maps your infrastructure assets and their relationships into an intuitive graph database, empowering you to proactively identify and mitigate security risks.** ([Original Repository](https://github.com/cartography-cncf/cartography))

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/cartography-cncf/cartography/badge)](https://scorecard.dev/viewer/?uri=github.com/cartography-cncf/cartography)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9637/badge)](https://www.bestpractices.dev/projects/9637)
![build](https://github.com/cartography-cncf/cartography/actions/workflows/publish-to-ghcr-and-pypi.yml/badge.svg)

## Key Features

*   **Comprehensive Asset Mapping:**  Consolidates infrastructure assets from various platforms.
*   **Relationship Visualization:**  Provides a clear, visual representation of relationships between assets using a Neo4j graph database.
*   **Security Risk Analysis:** Enables identification of hidden dependencies and potential attack paths.
*   **Extensible & Customizable:**  Easily integrates with new platforms and data sources.
*   **Flexible Exploration:** Supports both manual exploration via a web frontend and automated API access.
*   **Open Source:** Free to use and contribute to.

## Why Use Cartography?

Cartography offers a unique approach to understanding your infrastructure. It excels at revealing hidden dependencies, making it easier to validate security assumptions and proactively address risks.

*   **For Security Teams:** Identify potential attack paths and improve security posture.
*   **For Service Owners:** Generate asset reports and understand dependencies.
*   **For Red Teamers:** Discover attack paths for penetration testing.
*   **For Blue Teamers:**  Identify areas for security improvement.

Learn more about the story behind Cartography in our [presentation at BSidesSF 2019](https://www.youtube.com/watch?v=ZukUmZSKSek).

## Supported Platforms

Cartography currently supports a wide range of platforms, with new integrations constantly being added:

*   [Airbyte](https://cartography-cncf.github.io/cartography/modules/airbyte/index.html)
*   [Amazon Web Services](https://cartography-cncf.github.io/cartography/modules/aws/index.html)
*   [Anthropic](https://cartography-cncf.github.io/cartography/modules/anthropic/index.html)
*   [BigFix](https://cartography-cncf.github.io/cartography/modules/bigfix/index.html)
*   [Cloudflare](https://cartography-cncf.github.io/cartography/modules/cloudflare/index.html)
*   [Crowdstrike Falcon](https://cartography-cncf.github.io/cartography/modules/crowdstrike/index.html)
*   [DigitalOcean](https://cartography-cncf.github.io/cartography/modules/digitalocean/index.html)
*   [Duo](https://cartography-cncf.github.io/cartography/modules/duo/index.html)
*   [GitHub](https://cartography-cncf.github.io/cartography/modules/github/index.html)
*   [Google Cloud Platform](https://cartography-cncf.github.io/cartography/modules/gcp/index.html)
*   [Google GSuite](https://cartography-cncf.github.io/cartography/modules/gsuite/index.html)
*   [Kandji](https://cartography-cncf.github.io/cartography/modules/kandji/index.html)
*   [Keycloak](https://cartography-cncf.github.io/cartography/modules/keycloak/index.html)
*   [Kubernetes](https://cartography-cncf.github.io/cartography/modules/kubernetes/index.html)
*   [Lastpass](https://cartography-cncf.github.io/cartography/modules/lastpass/index.html)
*   [Microsoft Azure](https://cartography-cncf.github.io/cartography/modules/azure/index.html)
*   [Microsoft Entra ID](https://cartography-cncf.github.io/cartography/modules/entra/index.html)
*   [NIST CVE](https://cartography-cncf.github.io/cartography/modules/cve/index.html)
*   [Okta](https://cartography-cncf.github.io/cartography/modules/okta/index.html)
*   [OpenAI](https://cartography-cncf.github.io/cartography/modules/openai/index.html)
*   [Oracle Cloud Infrastructure](https://cartography-cncf.github.io/cartography/modules/oci/index.html)
*   [PagerDuty](https://cartography-cncf.github.io/cartography/modules/pagerduty/index.html)
*   [Scaleway](https://cartography-cncf.github.io/cartography/modules/scaleway/index.html)
*   [SentinelOne](https://cartography-cncf.github.io/cartography/modules/sentinelone/index.html)
*   [SnipeIT](https://cartography-cncf.github.io/cartography/modules/snipeit/index.html)
*   [Tailscale](https://cartography-cncf.github.io/cartography/modules/tailscale/index.html)
*   [Trivy Scanner](https://cartography-cncf.github.io/cartography/modules/trivy/index.html)

## Installation & Configuration

*   **Getting Started:** [Installation Guide](https://cartography-cncf.github.io/cartography/install.html)
*   **Production Setup:** [Production Recommendations](https://cartography-cncf.github.io/cartography/ops.html)

## Usage

### Querying the Database

Leverage the power of your infrastructure graph through our [querying tutorial](https://cartography-cncf.github.io/cartography/usage/tutorial.html) and reference our [data schema](https://cartography-cncf.github.io/cartography/usage/schema.html).

### Building Applications

Build applications and data pipelines using Cartography.  See our guide on [applications](https://cartography-cncf.github.io/cartography/usage/applications.html).

## Documentation

*   [Full Documentation](https://cartography-cncf.github.io/cartography/)

## Community

*   **Slack:** Join the CNCF Slack workspace [here](https://communityinviter.com/apps/cloud-native/cncf) and then the `#cartography` channel.
*   **Community Meetings:**  Attend our [monthly community meeting](https://zoom-lfx.platform.linuxfoundation.org/meetings/cartography?view=week) (Meeting minutes: [here](https://docs.google.com/document/d/1VyRKmB0dpX185I15BmNJZpfAJ_Ooobwz0U1WIhjDxvw); Recorded videos: [here](https://www.youtube.com/playlist?list=PLMga2YJvAGzidUWJB_fnG7EHI4wsDDsE1)).

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Contributing

We welcome contributions! Please follow the [CNCF code of conduct](https://github.com/cncf/foundation/blob/main/code-of-conduct.md).

*   **Report bugs/request features:** Submit a [GitHub issue](https://github.com/cartography-cncf/cartography/issues).
*   **Discussions:** Discuss larger topics in [GitHub Discussions](https://github.com/cartography-cncf/cartography/discussions).
*   **Developer Documentation:** [Developer Guide](https://cartography-cncf.github.io/cartography/dev/developer-guide.html)

## Who Uses Cartography?

*   [Lyft](https://www.lyft.com)
*   [Thought Machine](https://thoughtmachine.net/)
*   [MessageBird](https://messagebird.com)
*   [Cloudanix](https://www.cloudanix.com/)
*   [Corelight](https://www.corelight.com/)
*   [SubImage](https://subimage.io)
*   {Your company here} :-)

(If your organization uses Cartography, please submit a PR to add your company!)

---

Cartography is a [Cloud Native Computing Foundation](https://www.cncf.io/) sandbox project.
<div style="background-color: white; display: inline-block; padding: 10px;">
  <img src="docs/root/images/cncf-color.png" alt="CNCF Logo" width="200">
</div>