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

# authentik: Open-Source Identity Provider for Modern Applications

authentik is an open-source Identity Provider (IdP) that empowers you to manage user identities and access securely, with a focus on flexibility and versatility.  [Explore the authentik project on GitHub](https://github.com/goauthentik/authentik).

## Key Features

*   **Versatile Protocol Support:**  Supports a wide range of protocols, including  SAML, OIDC, LDAP, and more, for seamless integration with various applications.
*   **Flexible and Extensible:**  Designed for adaptability, allowing you to customize and extend its functionality to fit your specific needs.
*   **Self-Hosted Identity Management:**  Offers a self-hosted solution, giving you complete control over your identity and access management infrastructure.
*   **Enterprise-Grade Capabilities:** Suitable for large-scale deployments, providing features for both employee and B2B2C use cases, and can replace legacy IdPs.
*   **Open Source and Community Driven:** Benefit from the transparency and collaborative nature of open-source software.

## Installation

### Docker Compose (Recommended for small/test setups)

Follow the instructions in the [documentation](https://goauthentik.io/docs/installation/docker-compose/?utm_source=github).

### Kubernetes (for larger setups)

Use the Helm Chart available [here](https://github.com/goauthentik/helm).  Detailed instructions are available [here](https://goauthentik.io/docs/installation/kubernetes/?utm_source=github).

## Screenshots

| Light                                                       | Dark                                                       |
| ----------------------------------------------------------- | ---------------------------------------------------------- |
| ![](https://docs.goauthentik.io/img/screen_apps_light.jpg)  | ![](https://docs.goauthentik.io/img/screen_apps_dark.jpg)  |
| ![](https://docs.goauthentik.io/img/screen_admin_light.jpg) | ![](https://docs.goauthentik.io/img/screen_admin_dark.jpg) |

## Development

Refer to the [Developer Documentation](https://docs.goauthentik.io/docs/developer-docs/?utm_source=github) for information on contributing and developing authentik.

## Security

Review the [SECURITY.md](SECURITY.md) file for security considerations.

## Adoption and Contributions

If your organization uses authentik, we'd love to feature your logo!  Contact us at hello@goauthentik.io or open a GitHub Issue/PR.  For details on contributing to authentik, consult our [contribution guide](https://docs.goauthentik.io/docs/developer-docs?utm_source=github).