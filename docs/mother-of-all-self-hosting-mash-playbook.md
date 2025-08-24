# Mother-of-All-Self-Hosting (MASH) Playbook: Your Ultimate Self-Hosting Solution

**MASH** (Mother-of-All-Self-Hosting) is an Ansible playbook designed to simplify and streamline self-hosting by deploying and managing various services as Docker containers on your own server. ([View the original repo](https://github.com/mother-of-all-self-hosting/mash-playbook))

This project offers a comprehensive and efficient way to manage your self-hosted services, providing a predictable, up-to-date setup across multiple supported Linux distributions and CPU architectures. 

## Key Features

*   **Simplified Self-Hosting:** Manage multiple self-hosted services from a single Ansible playbook, eliminating the need to juggle multiple playbooks.
*   **Containerized Deployment:** Deploy services within Docker containers for consistent and isolated environments.
*   **Wide Service Support:** Supports a growing list of [FOSS](https://en.wikipedia.org/wiki/Free_and_open-source_software) services.  Explore the [full list of supported services](docs/supported-services.md).
*   **Automated Installation & Updates:** Leverage Ansible for automated installation, upgrades, and maintenance tasks. See our [Ansible guide](docs/ansible.md).
*   **Easy Service Addition:** Quickly experiment with new services with minimal configuration.
*   **Shared Service Management:** Centralized management of shared services like databases and reverse proxies, ensuring consistency and ease of maintenance.
*   **Simplified Backups:**  Everything lives together, making backups easy.

## Supported Services

[See the full list of supported services here](docs/supported-services.md).

## Installation

To configure and install services on your server, follow the [installation guide in the docs/ directory](docs/README.md).

## Changelog

Stay up-to-date with the latest changes and improvements by referring to the [changelog](CHANGELOG.md).

## Support

*   **Matrix Room:**  Join the community for support and discussions: [#mash-playbook:devture.com](https://matrixrooms.info/room/mash-playbook:devture.com).  Get started with a public server like `matrix.org` or self-host using [matrix-docker-ansible-deploy](https://github.com/spantaleev/matrix-docker-ansible-deploy).
*   **GitHub Issues:** Report bugs and request features through the [GitHub issues](https://github.com/mother-of-all-self-hosting/mash-playbook/issues).

## Why MASH?

MASH simplifies self-hosting by combining multiple service-specific Ansible playbooks into one. This reduces duplication, streamlines management, and enables easy experimentation with new services. It provides a high-quality, reliable, and integrated solution for your self-hosted needs.

**This playbook is designed to be your all-in-one toolkit for self-hosting services in a clean and reliable way.**