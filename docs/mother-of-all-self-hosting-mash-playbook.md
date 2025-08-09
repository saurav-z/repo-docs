[![Support room on Matrix](https://img.shields.io/matrix/mash-playbook:devture.com.svg?label=%23mash-playbook%3Adevture.com&logo=matrix&style=for-the-badge&server_fqdn=matrix.devture.com&fetchMode=summary)](https://matrixrooms.info/room/mash-playbook:devture.com)
[![donate](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/mother-of-all-self-hosting/donate)

# MASH Playbook: Your All-in-One Solution for Self-Hosting with Ansible

**MASH** (Mother-of-All-Self-Hosting) is an Ansible playbook designed to simplify and streamline self-hosting, allowing you to easily deploy and manage a wide range of services within Docker containers on your own server. **[View the original repository](https://github.com/mother-of-all-self-hosting/mash-playbook).**

## Key Features

*   **Simplified Self-Hosting:** Easily deploy and manage various services with a single Ansible playbook.
*   **Docker Containerization:** Utilize Docker for consistent, predictable, and up-to-date service setups across different Linux distributions and CPU architectures.
*   **Extensive Service Support:** Supports a [growing list of services](docs/supported-services.md), with a focus on Free and Open Source Software (FOSS).
*   **Automated Installation & Upgrades:** Leverage Ansible for automated installation, upgrades, and maintenance tasks.
*   **Centralized Management:** Manage shared services like databases and reverse proxies in one place, simplifying configuration and maintenance.
*   **Easy Experimentation:** Quickly try out new self-hosted services with minimal configuration.
*   **Backup Integration:** Simplified backup process due to everything residing in a single, organized playbook.

## Supported Services

Explore the [full list of supported services](docs/supported-services.md) to find the perfect tools for your self-hosted setup.

## Getting Started

To install and configure services on your server, follow the detailed instructions in the [README within the docs directory](docs/README.md).

## Changelog

Stay informed about updates and backward-incompatible changes by reviewing the [CHANGELOG.md](CHANGELOG.md) when updating the playbook.

## Support and Community

*   **Matrix Room:** Connect with the community in the `#mash-playbook:devture.com` [Matrix room](https://matrixrooms.info/room/mash-playbook:devture.com). You can join using a public server like `matrix.org` or self-host your own Matrix instance using [matrix-docker-ansible-deploy](https://github.com/spantaleev/matrix-docker-ansible-deploy).
*   **GitHub Issues:** Report issues and contribute to the project via [GitHub issues](https://github.com/mother-of-all-self-hosting/mash-playbook/issues).

## Why MASH?

MASH was created to consolidate the management of multiple Ansible playbooks for various services, reducing duplication of effort and making it easier to manage your self-hosted infrastructure. It allows you to experiment with new services with ease, maintain shared services in one place, and streamline your backup process.