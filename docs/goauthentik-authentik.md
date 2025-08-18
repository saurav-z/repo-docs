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

authentik is a versatile, open-source identity provider (IdP) offering flexible authentication and authorization, designed to streamline access management for your applications.  **[Explore the authentik GitHub repository](https://github.com/goauthentik/authentik).**

## Key Features

*   **Open-Source & Self-Hosted:** Gain full control over your identity infrastructure.
*   **Flexible Protocol Support:** Supports a wide range of authentication protocols.
*   **User-Friendly Interface:**  Intuitive admin and user dashboards.
*   **Enterprise-Grade Capabilities:**  Offers features suitable for large-scale deployments and B2B2C scenarios.
*   **Docker & Kubernetes Ready:** Easy installation via Docker Compose and Helm charts for Kubernetes.

## Use Cases

*   **Centralized Authentication:** Manage user identities and access control in one place.
*   **Multi-Factor Authentication (MFA):**  Enhance security with various MFA methods.
*   **SSO (Single Sign-On):**  Enable seamless access to multiple applications.
*   **Replace Legacy IdPs:** A cost-effective and feature-rich alternative to solutions like Okta, Auth0, and Entra ID.

## Installation

### Docker Compose (Recommended for small/test setups)

Refer to the [authentik documentation](https://goauthentik.io/docs/installation/docker-compose/?utm_source=github) for detailed instructions.

### Kubernetes (for larger setups)

Utilize the provided Helm Chart, documented [here](https://goauthentik.io/docs/installation/kubernetes/?utm_source=github).  The Helm chart is available at [https://github.com/goauthentik/helm](https://github.com/goauthentik/helm).

## Screenshots

| Light                                                       | Dark                                                       |
| ----------------------------------------------------------- | ---------------------------------------------------------- |
| ![](https://docs.goauthentik.io/img/screen_apps_light.jpg)  | ![](https://docs.goauthentik.io/img/screen_apps_dark.jpg)  |
| ![](https://docs.goauthentik.io/img/screen_admin_light.jpg) | ![](https://docs.goauthentik.io/img/screen_admin_dark.jpg) |

## Development

Review the [Developer Documentation](https://docs.goauthentik.io/docs/developer-docs/?utm_source=github) for information on contributing to the project.

## Security

Learn more about security considerations in [SECURITY.md](SECURITY.md).

## Adoption and Contributions

If your organization uses authentik, we'd love to feature your logo!  Please contact us at hello@goauthentik.io or open a GitHub Issue/PR.  For information on contributing, consult our [contribution guide](https://docs.goauthentik.io/docs/developer-docs?utm_source=github).