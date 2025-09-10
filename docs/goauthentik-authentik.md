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

# authentik: Open-Source Identity Provider for Modern SSO

authentik is a powerful, open-source Identity Provider (IdP) that allows you to easily implement Single Sign-On (SSO) and secure access for your applications and infrastructure.  [Explore the authentik repository on GitHub](https://github.com/goauthentik/authentik).

## Key Features

*   **Comprehensive Protocol Support:** authentik supports a wide range of authentication protocols, including SAML, OAuth2/OIDC, LDAP, and RADIUS, offering flexibility for various use cases.
*   **Self-Hosting:** Designed for self-hosting, from small labs to large production clusters, giving you complete control over your identity infrastructure.
*   **Enterprise-Grade Capabilities:** The enterprise offering provides advanced features to securely replace existing IdPs such as Okta, Auth0, Entra ID, and Ping Identity.
*   **Flexible Deployment Options:** Deploy authentik using Docker Compose, Kubernetes (Helm Chart), AWS CloudFormation, or through the DigitalOcean Marketplace for easy setup and scalability.
*   **User-Friendly Interface:**  Intuitive web interfaces for both administrators and end-users, ensuring a seamless experience.

## Installation

Choose the deployment method that best suits your needs:

*   **Docker Compose:** Recommended for small setups and testing. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/docker-compose/).
*   **Kubernetes (Helm Chart):**  Recommended for larger deployments. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/kubernetes/) and the Helm chart [repository](https://github.com/goauthentik/helm).
*   **AWS CloudFormation:** Deploy on AWS using our official templates. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/aws/).
*   **DigitalOcean Marketplace:** One-click deployment via the official Marketplace app. See the [app listing](https://marketplace.digitalocean.com/apps/authentik).

## Screenshots

| Light                                                       | Dark                                                       |
| ----------------------------------------------------------- | ---------------------------------------------------------- |
| ![](https://docs.goauthentik.io/img/screen_apps_light.jpg)  | ![](https://docs.goauthentik.io/img/screen_apps_dark.jpg)  |
| ![](https://docs.goauthentik.io/img/screen_admin_light.jpg) | ![](https://docs.goauthentik.io/img/screen_admin_dark.jpg) |

## Development and Contributions

Learn how to contribute to authentik by exploring the [Developer Documentation](https://docs.goauthentik.io/docs/developer-docs/).  You'll find information on:

*   Setting up local build environments
*   Testing your contributions
*   The contribution process

## Security

For security information, please review the [SECURITY.md](SECURITY.md) file.

## Adoption

We'd love to feature your organization! If you are using authentik, please share your story by emailing us at [hello@goauthentik.io](mailto:hello@goauthentik.io) or open a GitHub Issue/PR.

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey?style=for-the-badge)](website/LICENSE)
[![authentik EE License](https://img.shields.io/badge/License-EE-orange?style=for-the-badge)](authentik/enterprise/LICENSE)
```
Key improvements and SEO considerations:

*   **Clear and Concise Hook:** The opening sentence directly addresses the core function of authentik and includes keywords for searchability.
*   **Keyword Optimization:**  Includes keywords like "Open-Source Identity Provider," "SSO," "SAML," "OAuth2/OIDC," and deployment options.
*   **Structured Headings:** Organizes the information with clear headings and subheadings for better readability and SEO.
*   **Bulleted Key Features:** Highlights the main benefits in a scannable format.
*   **Links Back to the Original Repo:**  Explicitly mentions where to find the source code, linking back to the original repository.
*   **Deployment Options:**  Provides better descriptions of installation methods and links to the relevant documentation.
*   **Concise Language:**  Rephrases some sentences for better clarity.
*   **Call to Action:** Encourages users to share their adoption stories.