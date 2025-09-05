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

# authentik: Open-Source Identity Provider for Secure SSO

authentik is a powerful, open-source Identity Provider (IdP) solution designed for modern Single Sign-On (SSO) needs, perfect for both small deployments and large-scale production environments.  Explore the original repository on [GitHub](https://github.com/goauthentik/authentik).

## Key Features

*   **Comprehensive Protocol Support:** Integrates with SAML, OAuth2/OIDC, LDAP, RADIUS, and more.
*   **Self-Hosting:** Designed for self-hosting, providing complete control over your identity infrastructure.
*   **Scalable Architecture:** Suitable for deployments ranging from small labs to large production clusters.
*   **Enterprise-Ready:** Offers an enterprise offering to replace existing IdPs like Okta, Auth0, Entra ID, and Ping Identity.
*   **Multiple Installation Options:** Supports Docker Compose, Kubernetes (Helm Chart), AWS CloudFormation, and DigitalOcean Marketplace.

## Installation

Choose your preferred installation method:

*   **Docker Compose:** Ideal for small setups and testing. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/docker-compose/).
*   **Kubernetes (Helm Chart):** Recommended for larger deployments. Refer to the [documentation](https://docs.goauthentik.io/docs/install-config/install/kubernetes/) and the Helm chart [repository](https://github.com/goauthentik/helm).
*   **AWS CloudFormation:** Deploy on AWS using our official templates. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/aws/).
*   **DigitalOcean Marketplace:** One-click deployment available via the official Marketplace app. See the [app listing](https://marketplace.digitalocean.com/apps/authentik).

## Screenshots

| Light                                                       | Dark                                                       |
| ----------------------------------------------------------- | ---------------------------------------------------------- |
| ![](https://docs.goauthentik.io/img/screen_apps_light.jpg)  | ![](https://docs.goauthentik.io/img/screen_apps_dark.jpg)  |
| ![](https://docs.goauthentik.io/img/screen_admin_light.jpg) | ![](https://docs.goauthentik.io/img/screen_admin_dark.jpg) |

## Development and Contributions

Learn how to contribute and set up your local development environment. Explore the [Developer Documentation](https://docs.goauthentik.io/docs/developer-docs/) for detailed information on:

*   Setting up local build environments
*   Testing your contributions
*   Contribution process

## Security

For security information, please review [SECURITY.md](SECURITY.md).

## Adoption

We'd love to hear how you are using authentik!  Share your story and have your logo featured by contacting us at [hello@goauthentik.io](mailto:hello@goauthentik.io) or open a GitHub Issue/PR!

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey?style=for-the-badge)](website/LICENSE)
[![authentik EE License](https://img.shields.io/badge/License-EE-orange?style=for-the-badge)](authentik/enterprise/LICENSE)