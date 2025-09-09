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

# authentik: Your Open-Source Identity Provider for Modern SSO

authentik is a powerful, open-source Identity Provider (IdP) designed for self-hosting, offering robust SSO capabilities for organizations of all sizes. Learn more on the [authentik GitHub](https://github.com/goauthentik/authentik)

## Key Features

*   **Comprehensive Protocol Support:** Authentik supports SAML, OAuth2/OIDC, LDAP, RADIUS, and more, providing versatile authentication options.
*   **Self-Hosting Focused:**  Designed for easy self-hosting, from small labs to large production clusters, giving you control over your identity infrastructure.
*   **Scalable Architecture:** Built to handle the demands of growing organizations, with options for Kubernetes deployments.
*   **Enterprise-Grade Capabilities:**  Seamlessly replace existing IdPs like Okta, Auth0, and Entra ID with authentik's enterprise offering for secure, large-scale identity management.
*   **Flexible Deployment Options:** Choose from Docker Compose, Kubernetes (Helm Chart), AWS CloudFormation, or DigitalOcean Marketplace for easy installation.

## Installation Guides

Choose the installation method that best suits your needs:

*   **Docker Compose:** Recommended for small setups and testing.  See the [documentation](https://docs.goauthentik.io/docs/install-config/install/docker-compose/).
*   **Kubernetes (Helm Chart):** Ideal for larger deployments. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/kubernetes/) and the Helm chart [repository](https://github.com/goauthentik/helm).
*   **AWS CloudFormation:** Deploy on AWS using our official templates. See the [documentation](https://docs.goauthentik.io/docs/install-config/install/aws/).
*   **DigitalOcean Marketplace:** Deploy with a single click. See the [app listing](https://marketplace.digitalocean.com/apps/authentik).

## Screenshots

| Light                                                       | Dark                                                       |
| ----------------------------------------------------------- | ---------------------------------------------------------- |
| ![](https://docs.goauthentik.io/img/screen_apps_light.jpg)  | ![](https://docs.goauthentik.io/img/screen_apps_dark.jpg)  |
| ![](https://docs.goauthentik.io/img/screen_admin_light.jpg) | ![](https://docs.goauthentik.io/img/screen_admin_dark.jpg) |

## Development and Contribution

Contribute to authentik and help improve the project!  See the [Developer Documentation](https://docs.goauthentik.io/docs/developer-docs/) for setting up your environment, testing contributions, and learning about the contribution process.

## Security

Security is a top priority. See [SECURITY.md](SECURITY.md) for details on security practices.

## Adoption

Are you using authentik? We'd love to hear about your experience!  Email us at [hello@goauthentik.io](mailto:hello@goauthentik.io) or open a GitHub Issue/PR to share your story.

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey?style=for-the-badge)](website/LICENSE)
[![authentik EE License](https://img.shields.io/badge/License-EE-orange?style=for-the-badge)](authentik/enterprise/LICENSE)