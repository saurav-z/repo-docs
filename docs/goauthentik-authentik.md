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

# authentik: Open-Source Identity Provider (IdP) for Modern SSO

authentik is a powerful, open-source Identity Provider (IdP) that simplifies modern Single Sign-On (SSO) for organizations of all sizes, offering secure access management and streamlined user authentication.  [Explore the authentik project on GitHub](https://github.com/goauthentik/authentik).

## Key Features

*   **Comprehensive Protocol Support:** Supports industry-standard protocols including SAML, OAuth2/OIDC, LDAP, and RADIUS, providing flexibility and broad compatibility.
*   **Self-Hosting Capabilities:** Designed for self-hosting, allowing for complete control over your identity infrastructure, ideal for various environments from small labs to large production clusters.
*   **Enterprise-Grade Solution:**  Offers an [enterprise offering](https://goauthentik.io/pricing) for robust, large-scale identity management, capable of replacing existing IdPs like Okta, Auth0, Entra ID, and Ping Identity.
*   **Flexible Deployment Options:** Provides multiple installation methods including Docker Compose, Kubernetes (Helm Chart), AWS CloudFormation, and DigitalOcean Marketplace for easy setup and scaling.

## Installation

Choose the installation method that best suits your needs:

*   **Docker Compose:** Recommended for small/test setups. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/docker-compose/).
*   **Kubernetes (Helm Chart):** Recommended for larger setups. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/kubernetes/) and the Helm chart [repository](https://github.com/goauthentik/helm).
*   **AWS CloudFormation:** Deploy on AWS using our official templates. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/aws/).
*   **DigitalOcean Marketplace:** One-click deployment via the official Marketplace app. See the [app listing](https://marketplace.digitalocean.com/apps/authentik).

## Screenshots

| Light                                                       | Dark                                                       |
| ----------------------------------------------------------- | ---------------------------------------------------------- |
| ![](https://docs.goauthentik.io/img/screen_apps_light.jpg)  | ![](https://docs.goauthentik.io/img/screen_apps_dark.jpg)  |
| ![](https://docs.goauthentik.io/img/screen_admin_light.jpg) | ![](https://docs.goauthentik.io/img/screen_admin_dark.jpg) |

## Development and Contributions

Contribute to the project! See the [Developer Documentation](https://docs.goauthentik.io/docs/developer-docs/) for information about setting up local build environments, testing your contributions, and our contribution process.

## Security

See the [SECURITY.md](SECURITY.md) file for details about security considerations.

## Adoption

We'd love to hear your story and feature your logo! If you are using authentik, email us at [hello@goauthentik.io](mailto:hello@goauthentik.io) or open a GitHub Issue/PR!

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey?style=for-the-badge)](website/LICENSE)
[![authentik EE License](https://img.shields.io/badge/License-EE-orange?style=for-the-badge)](authentik/enterprise/LICENSE)