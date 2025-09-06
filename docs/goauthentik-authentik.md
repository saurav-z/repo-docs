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

**authentik** is a powerful and versatile open-source Identity Provider (IdP) designed for seamless Single Sign-On (SSO) across various applications and services. Find out more on the [authentik GitHub](https://github.com/goauthentik/authentik).

## Key Features

*   **Comprehensive Protocol Support:** Supports SAML, OAuth2/OIDC, LDAP, RADIUS, and more, offering broad compatibility.
*   **Self-Hosting:** Designed for self-hosting, from small labs to large production clusters, providing complete control over your identity infrastructure.
*   **Scalable Architecture:** Ready to scale with your needs, making it suitable for various deployments.
*   **Enterprise-Grade Capabilities:** Offers an enterprise offering for robust identity management, including features for larger organizations.
*   **Flexible Deployment Options:** Install with Docker Compose, Kubernetes (Helm Chart), AWS CloudFormation, or DigitalOcean Marketplace for easy setup.

## Installation

Choose the best installation method for your needs:

*   **Docker Compose:** Recommended for small and test setups. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/docker-compose/).
*   **Kubernetes (Helm Chart):** Recommended for larger setups. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/kubernetes/) and the Helm chart [repository](https://github.com/goauthentik/helm).
*   **AWS CloudFormation:** Deploy on AWS using official templates. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/aws/).
*   **DigitalOcean Marketplace:** One-click deployment via the official Marketplace app. See the [app listing](https://marketplace.digitalocean.com/apps/authentik).

## Screenshots

| Light                                                       | Dark                                                       |
| ----------------------------------------------------------- | ---------------------------------------------------------- |
| ![](https://docs.goauthentik.io/img/screen_apps_light.jpg)  | ![](https://docs.goauthentik.io/img/screen_apps_dark.jpg)  |
| ![](https://docs.goauthentik.io/img/screen_admin_light.jpg) | ![](https://docs.goauthentik.io/img/screen_admin_dark.jpg) |

## Development and Contribution

Interested in contributing? Explore the [Developer Documentation](https://docs.goauthentik.io/docs/developer-docs/) for setting up local build environments, testing, and understanding the contribution process.

## Security

Security is a priority. Review our [SECURITY.md](SECURITY.md) for more details.

## Adoption

We love to see our users! If you're using authentik, share your story by emailing us at [hello@goauthentik.io](mailto:hello@goauthentik.io) or opening a GitHub Issue/PR!

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey?style=for-the-badge)](website/LICENSE)
[![authentik EE License](https://img.shields.io/badge/License-EE-orange?style=for-the-badge)](authentik/enterprise/LICENSE)