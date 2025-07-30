[![Support room on Matrix](https://img.shields.io/matrix/mash-playbook:devture.com.svg?label=%23mash-playbook%3Adevture.com&logo=matrix&style=for-the-badge&server_fqdn=matrix.devture.com&fetchMode=summary)](https://matrixrooms.info/room/mash-playbook:devture.com) [![donate](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/mother-of-all-self-hosting/donate)

# MASH: The Ultimate Ansible Playbook for Self-Hosting

**MASH (Mother-of-All-Self-Hosting) is an Ansible playbook designed to simplify self-hosting by deploying a wide array of services as Docker containers on your own server.**  This provides a streamlined, predictable, and easily updated environment for all your self-hosted needs.

[**Visit the original repository on GitHub**](https://github.com/mother-of-all-self-hosting/mash-playbook)

## Key Features

*   **Simplified Self-Hosting:** Manage multiple services with a single Ansible playbook.
*   **Docker Containerization:** Ensures consistent and up-to-date service deployments across different Linux distributions and CPU architectures.
*   **Wide Range of Supported Services:**  Easily host a [growing number of services](docs/supported-services.md), with ongoing additions of Free and Open Source Software (FOSS).
*   **Automated Installation and Upgrades:**  Leverage Ansible for automated configuration, installation, and maintenance tasks.
*   **Easy Service Integration:** Built-in support for common services like PostgreSQL and Traefik, simplifying setup and management.
*   **Centralized Backups:**  Simplified backup processes because all services share a common data path and PostgreSQL instance.

## Supported Services

See the [full list of supported services here](docs/supported-services.md).

## Getting Started

To configure and install services on your server, follow the instructions in the [README](docs/README.md) within the `docs/` directory.

## Changelog

Stay up-to-date with the latest changes and updates by reviewing the [CHANGELOG.md](CHANGELOG.md).

## Support and Community

*   **Matrix:** Join the community in the [#mash-playbook:devture.com](https://matrixrooms.info/room/mash-playbook:devture.com) Matrix room for support and discussions.
*   **GitHub Issues:** Report bugs, request features, or ask questions on the [GitHub Issues page](https://github.com/mother-of-all-self-hosting/mash-playbook/issues).

## Why MASH? The Philosophy Behind the Playbook

MASH was created to solve the common challenges of managing multiple self-hosted services.  By combining a variety of services into a single, cohesive playbook, MASH offers:

*   **Reduced Complexity:** Eliminate the need to juggle multiple, independent Ansible playbooks.
*   **Ease of Experimentation:** Easily try out new services with minimal configuration.
*   **Consistent Quality:** Leverage a well-maintained and trusted playbook.
*   **Simplified Maintenance:** Share common services (e.g., PostgreSQL) and streamline backup procedures.
*   **Flexibility:** Use as many servers as you need to host your services.

The goal is to provide a robust and user-friendly tool that empowers you to create your own self-hosted ecosystem, with the flexibility to tailor it to your specific needs.