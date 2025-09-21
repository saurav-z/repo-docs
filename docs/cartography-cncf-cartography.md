<!-- Improved README for Cartography -->

![Cartography Logo](docs/root/images/logo-horizontal.png)

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/cartography-cncf/cartography/badge)](https://scorecard.dev/viewer/?uri=github.com/cartography-cncf/cartography)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9637/badge)](https://www.bestpractices.dev/projects/9637)
![Build Status](https://github.com/cartography-cncf/cartography/actions/workflows/publish-to-ghcr-and-pypi.yml/badge.svg)

## Cartography: Uncover and Visualize Your Infrastructure's Hidden Relationships

Cartography is a powerful open-source tool that builds an intuitive graph view of your infrastructure, enabling security teams to discover and understand complex relationships between assets.  For more information, check out the original repo at [https://github.com/cartography-cncf/cartography](https://github.com/cartography-cncf/cartography).

### Key Features

*   **Automated Asset Discovery:**  Automatically ingests data from various cloud providers, SaaS platforms, and on-premise systems.
*   **Graph Database Visualization:**  Stores data in a Neo4j graph database for intuitive visualization of relationships.
*   **Relationship Mapping:**  Exposes hidden dependencies and relationships between infrastructure components.
*   **Security Focused:**  Helps identify potential attack paths, security risks, and areas for improvement.
*   **Extensible:**  Supports custom plugins to integrate with new platforms and data sources.
*   **Web Interface and APIs:**  Provides both a user-friendly web interface for exploration and APIs for automation.
*   **Extensive Platform Support:**  Includes support for many popular cloud and SaaS platforms like AWS, GCP, Azure, GitHub, and more.

![Visualization of RDS and AWS nodes](docs/root/images/accountsandrds.png)

### Why Choose Cartography?

*   **Comprehensive Visibility:** Gain a holistic view of your infrastructure and its interdependencies.
*   **Risk Assessment:**  Identify vulnerabilities and assess potential attack vectors.
*   **Improved Security Posture:**  Enhance security practices and streamline incident response.
*   **Automation and Integration:**  Build automation workflows and integrate with other security tools.
*   **Open Source & Community Driven:** Benefit from the collaborative development and active community.

### Supported Platforms

Cartography supports a wide range of platforms. For details, visit each module's documentation:

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

### Getting Started

*   **Installation:**  Follow the [installation guide](https://cartography-cncf.github.io/cartography/install.html) to set up Cartography.
*   **Production Setup:**  Consult the [operational recommendations](https://cartography-cncf.github.io/cartography/ops.html) for production deployments.
*   **Querying:** Explore your data using the [querying tutorial](https://cartography-cncf.github.io/cartography/usage/tutorial.html) and [data schema](https://cartography-cncf.github.io/cartography/usage/schema.html).
*   **Applications:** Learn about building applications with Cartography in the [applications documentation](https://cartography-cncf.github.io/cartography/usage/applications.html).

### Documentation

Comprehensive documentation is available [here](https://cartography-cncf.github.io/cartography/).

### Community

*   **Slack:** Join the CNCF Slack workspace [here](https://communityinviter.com/apps/cloud-native/cncf) and then join the `#cartography` channel.
*   **Community Meetings:** Participate in the monthly community meeting - [meeting details](https://zoom-lfx.platform.linuxfoundation.org/meetings/cartography?view=week).  Meeting minutes are [here](https://docs.google.com/document/d/1VyRKmB0dpX185I15BmNJZpfAJ_Ooobwz0U1WIhjDxvw).
*   **Meeting Recordings:**  View recordings of previous meetings [here](https://www.youtube.com/playlist?list=PLMga2YJvAGzidUWJB_fnG7EHI4wsDDsE1).

### License

This project is licensed under the [Apache 2.0 License](LICENSE).

### Contributing

Contributions are welcome! Review the [developer documentation](https://cartography-cncf.github.io/cartography/dev/developer-guide.html) and follow the [CNCF code of conduct](https://github.com/cncf/foundation/blob/main/code-of-conduct.md).

*   **Bug Reports & Feature Requests:**  Submit issues on GitHub.
*   **Discussions:** Engage in discussions via [GitHub Discussions](https://github.com/cartography-cncf/cartography/discussions).

### Who Uses Cartography?

*   Lyft
*   Thought Machine
*   MessageBird
*   Cloudanix
*   Corelight
*   SubImage
*   {Your company here} :-)

Add your company to the list by submitting a pull request!

---

Cartography is a [Cloud Native Computing Foundation](https://www.cncf.io/) sandbox project.<br>
<div style="background-color: white; display: inline-block; padding: 10px;">
  <img src="docs/root/images/cncf-color.png" alt="CNCF Logo" width="200">
</div>