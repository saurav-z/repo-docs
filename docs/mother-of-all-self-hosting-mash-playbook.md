# Self-Host Your Digital Life with the Mother-of-All-Self-Hosting Playbook

**Simplify your self-hosting journey and take control of your data with the Mother-of-All-Self-Hosting (MASH) Ansible playbook, a comprehensive solution for deploying and managing self-hosted services using Docker containers.** ([View the original repository](https://github.com/mother-of-all-self-hosting/mash-playbook))

MASH streamlines self-hosting by automating the deployment and management of numerous services, all within Docker containers, ensuring a consistent and easily updatable setup across various Linux distributions and CPU architectures.

**Key Features:**

*   **Simplified Deployment:** Automates the installation and configuration of self-hosted services.
*   **Docker-Based:** Leverages Docker containers for consistent and isolated service environments.
*   **Comprehensive Service Support:** Supports a wide range of popular services, with continuous expansion. [View the list of supported services](docs/supported-services.md).
*   **Ansible Automation:**  Automates installation, upgrades, and maintenance tasks using Ansible.
*   **Easy Updates:**  Stay up-to-date with the latest features and security patches via the CHANGELOG.md.
*   **Centralized Management:** Reduces the need to manage multiple playbooks, simplifying your self-hosting infrastructure.
*   **Shared Resources:** Manages shared services like databases and reverse proxies in a single, unified location.
*   **Easy Backups:** Simplifies the backup process by keeping all data within the same base path and Postgres instance.

## Supported Services

Explore the [full list of supported services](docs/supported-services.md) to see the wide array of applications you can self-host with MASH.

## Getting Started

To configure and install services on your server, follow the instructions in the [README](docs/README.md) within the `docs/` directory.

## Changes & Updates

Stay informed about changes and new features by consulting the [CHANGELOG.md](CHANGELOG.md) when updating the playbook.

## Support

*   **Matrix Room:** Connect with the community and get support in the Matrix room: [#mash-playbook:devture.com](https://matrixrooms.info/room/mash-playbook:devture.com).
*   **GitHub Issues:** Report issues and track progress on GitHub: [mother-of-all-self-hosting/mash-playbook/issues](https://github.com/mother-of-all-self-hosting/mash-playbook/issues)

## Why MASH?

MASH consolidates the functionality of multiple playbooks, providing a streamlined and efficient approach to self-hosting:

*   Avoids the complexity of managing numerous individual playbooks.
*   Simplifies the process of trying out new services with minimal configuration.
*   Provides a home for smaller applications that might not warrant their own dedicated playbooks.
*   Ensures consistency and quality across all supported services.
*   Centralizes the management of shared services such as databases and reverse proxies.
*   Facilitates easy and unified backups.

While MASH supports hosting a wide variety of services, you're not limited to a single server.  Utilize as many servers as needed for your infrastructure.