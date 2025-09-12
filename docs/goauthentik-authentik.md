<p align="center">
    <img src="https://goauthentik.io/img/icon_top_brand_colour.svg" height="150" alt="authentik logo">
</p>

---

[![Join Discord](https://img.shields.io/discord/809154715984199690?label=Discord&style=for-the-badge)](https://goauthentik.io/discord)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/goauthentik/authentik/ci-main.yml?branch=main&label=core%20build&style=for-the-badge)](https://github.com/goauthentik/authentik/actions/workflows/ci-main.yml)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/goauthentik/authentik/ci-outpost.yml?branch=main&label=outpost%20build&style=for-the-badge)](https://github.com/goauthentik/authentik/actions/workflows/ci-outpost.yml)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/goauthentik/authentik/ci-web.yml?branch=main&label=web%20build&style=for-the-badge)](https://github.com/goauthentik/authentik/actions/workflows/ci-web.yml)
[![Code Coverage](https://img.shields.io/codecov/c/gh/goauthentik/authentik?style=for-the-badge)](https://codecov.io/gh/goauthentik/authentik)
![Docker pulls](https://img.shields.io/docker/pulls/authentik/server.svg?style=for-the-badge)
![Latest version](https://img.shields.io/docker/v/authentik/server?sort=semver&style=for-the-badge)
[![](https://img.shields.io/badge/Help%20translate-transifex-blue?style=for-the-badge)](https://www.transifex.com/authentik/authentik/)

# authentik: Open-Source Identity Provider (IdP) for Secure SSO

authentik is a powerful, open-source Identity Provider (IdP) that provides secure Single Sign-On (SSO) for modern applications, ideal for self-hosting and replacing existing IdPs like Okta, Auth0, and Entra ID. Check out the [original repo](https://github.com/goauthentik/authentik) for the full details.

## Key Features

*   **Comprehensive Protocol Support:** Offers robust support for SAML, OAuth2/OIDC, LDAP, and RADIUS, ensuring compatibility with a wide range of applications.
*   **Self-Hosting Flexibility:** Designed for easy self-hosting, accommodating setups from small labs to large production clusters.
*   **Enterprise-Grade Capabilities:** The enterprise offering provides advanced features for organizations seeking robust, large-scale identity management.
*   **Multiple Deployment Options:** Supports Docker Compose, Kubernetes (Helm Chart), AWS CloudFormation, and DigitalOcean Marketplace for flexible installation.

## Installation

Choose the installation method that best suits your needs:

*   **Docker Compose:** Recommended for small setups and testing.  Refer to the [documentation](https://docs.goauthentik.io/docs/install-config/install/docker-compose/).
*   **Kubernetes (Helm Chart):** Recommended for larger deployments. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/kubernetes/) and the Helm chart [repository](https://github.com/goauthentik/helm).
*   **AWS CloudFormation:** Deploy on AWS using official templates.  Consult the [documentation](https://docs.goauthentik.io/docs/install-config/install/aws/).
*   **DigitalOcean Marketplace:**  One-click deployment is available via the official Marketplace app. See the [app listing](https://marketplace.digitalocean.com/apps/authentik).

## Screenshots

| Light                                                       | Dark                                                       |
| ----------------------------------------------------------- | ---------------------------------------------------------- |
| ![](https://docs.goauthentik.io/img/screen_apps_light.jpg)  | ![](https://docs.goauthentik.io/img/screen_apps_dark.jpg)  |
| ![](https://docs.goauthentik.io/img/screen_admin_light.jpg) | ![](https://docs.goauthentik.io/img/screen_admin_dark.jpg) |

## Development and Contributions

For information on setting up local build environments, testing your contributions, and our contribution process, see the [Developer Documentation](https://docs.goauthentik.io/docs/developer-docs/).

## Security

Review our security practices in [SECURITY.md](SECURITY.md).

## Adoption

If you are using authentik, we'd love to hear your story. Share your experience and potentially feature your logo! Email us at [hello@goauthentik.io](mailto:hello@goauthentik.io) or open a GitHub Issue/PR.

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey?style=for-the-badge)](website/LICENSE)
[![authentik EE License](https://img.shields.io/badge/License-EE-orange?style=for-the-badge)](authentik/enterprise/LICENSE)
```
Key improvements and SEO considerations:

*   **Clear Headline:**  Uses a clear, concise headline.
*   **Hook:** A single-sentence opening to capture attention, emphasizing the core value proposition.
*   **Keywords:** Includes relevant keywords like "Identity Provider," "IdP," "SSO," "SAML," "OAuth2," "self-hosting," "open-source," and names of competitors like Okta, Auth0, and Entra ID.
*   **Bulleted Key Features:**  Highlights the most important features in an easy-to-scan format.
*   **Installation Section:**  Organized with clear headings and links.
*   **Concise Language:**  Avoids unnecessary jargon.
*   **Call to Action (Adoption):**  Encourages community engagement.
*   **Link Back to Original Repo:**  Clearly states where to find more details.