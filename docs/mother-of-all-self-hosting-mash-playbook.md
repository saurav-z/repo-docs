[![Support room on Matrix](https://img.shields.io/matrix/mash-playbook:devture.com.svg?label=%23mash-playbook%3Adevture.com&logo=matrix&style=for-the-badge&server_fqdn=matrix.devture.com&fetchMode=summary)](https://matrixrooms.info/room/mash-playbook:devture.com) [![donate](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/mother-of-all-self-hosting/donate)

# Self-Host Your Dream Services with the Mother-of-All-Self-Hosting Playbook

**MASH** (Mother-of-All-Self-Hosting) is an Ansible playbook designed to simplify and streamline the process of self-hosting various services in Docker containers on your own server.

This playbook is designed to help you easily deploy and manage a wide array of self-hosted services, offering a predictable and up-to-date setup across multiple Linux distributions and CPU architectures.

**[See the original repository here](https://github.com/mother-of-all-self-hosting/mash-playbook)**

## Key Features

*   **Simplified Self-Hosting:** Easily deploy and manage numerous services within Docker containers.
*   **Containerized Deployment:** Ensures a consistent and reliable setup across different environments.
*   **Wide Service Support:** Supports a [growing list of services](docs/supported-services.md), including popular FOSS applications.
*   **Automated Updates & Maintenance:** Leverage Ansible for automated installation, upgrades, and maintenance tasks.
*   **Cross-Platform Compatibility:** Designed to work on multiple Linux distributions and CPU architectures.
*   **Centralized Management:** Manage all your self-hosted services from a single, unified playbook.
*   **Easy to Extend:** Add new services and customize your setup with ease.

## Supported Services

Explore the [full list of supported services](docs/supported-services.md) to discover what you can self-host. This list is constantly growing, with new services and features being added regularly.

## Getting Started

To configure and install services on your own server, follow the detailed instructions in the [README located in the docs/ directory](docs/README.md).

## Updates and Changelog

This playbook is actively maintained and updated. Review the [CHANGELOG.md](CHANGELOG.md) to stay informed about the latest changes, updates, and any potential backward-incompatible changes.

## Support and Community

*   **Matrix Room:** Join the community for support and discussions at [#mash-playbook:devture.com](https://matrixrooms.info/room/mash-playbook:devture.com).
*   **GitHub Issues:** Report bugs, request features, or ask questions via [GitHub issues](https://github.com/mother-of-all-self-hosting/mash-playbook/issues).

## Why MASH? The Philosophy

MASH aims to eliminate the complexity of managing multiple Ansible playbooks for self-hosting. It consolidates common dependencies (like databases and reverse proxies) and simplifies the process of trying out new services. With MASH, you can easily build and maintain your ideal self-hosted stack, leveraging the power of Docker and Ansible for a reliable, up-to-date, and easy-to-manage setup.