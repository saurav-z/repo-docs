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

authentik is a powerful open-source Identity Provider (IdP) that allows you to centrally manage user authentication and authorization for all your applications and services.  ([View on GitHub](https://github.com/goauthentik/authentik))

## Key Features

*   **Comprehensive Protocol Support:**  Supports SAML, OAuth2/OIDC, LDAP, RADIUS, and more, providing flexibility and compatibility with a wide range of applications.
*   **Self-Hosting Focused:** Designed for easy self-hosting, from small personal projects to large-scale production deployments.
*   **Modern SSO:**  Provides a centralized solution for Single Sign-On (SSO), improving user experience and security.
*   **Enterprise-Grade Capabilities:**  Offers an [enterprise offering](https://goauthentik.io/pricing) to replace existing IdPs like Okta, Auth0, and Entra ID for robust identity management.
*   **Flexible Deployment Options:**  Offers multiple installation methods, including Docker Compose, Kubernetes (Helm Chart), AWS CloudFormation, and a DigitalOcean Marketplace app.

## Installation

Choose the installation method that best suits your needs:

*   **Docker Compose:** Recommended for small setups and testing.  See the [documentation](https://docs.goauthentik.io/docs/install-config/install/docker-compose/).
*   **Kubernetes (Helm Chart):** Recommended for larger deployments.  See the [documentation](https://docs.goauthentik.io/docs/install-config/install/kubernetes/) and the Helm chart [repository](https://github.com/goauthentik/helm).
*   **AWS CloudFormation:** Deploy on AWS using our official templates. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/aws/).
*   **DigitalOcean Marketplace:** One-click deployment via the official Marketplace app. See the [app listing](https://marketplace.digitalocean.com/apps/authentik).

## Screenshots

| Light                                                       | Dark                                                       |
| ----------------------------------------------------------- | ---------------------------------------------------------- |
| ![](https://docs.goauthentik.io/img/screen_apps_light.jpg)  | ![](https://docs.goauthentik.io/img/screen_apps_dark.jpg)  |
| ![](https://docs.goauthentik.io/img/screen_admin_light.jpg) | ![](https://docs.goauthentik.io/img/screen_admin_dark.jpg) |

## Development and Contributions

Contribute to the project! See the [Developer Documentation](https://docs.goauthentik.io/docs/developer-docs/) for information on setting up your development environment, testing your contributions, and the contribution process.

## Security

Security is paramount. Please review our [SECURITY.md](SECURITY.md) document for details on security practices and reporting vulnerabilities.

## Adoption

We love to hear about users of authentik! If you're using authentik, please share your story by emailing us at [hello@goauthentik.io](mailto:hello@goauthentik.io) or opening a GitHub Issue/PR to be featured.

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey?style=for-the-badge)](website/LICENSE)
[![authentik EE License](https://img.shields.io/badge/License-EE-orange?style=for-the-badge)](authentik/enterprise/LICENSE)