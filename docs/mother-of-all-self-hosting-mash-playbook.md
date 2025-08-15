[![Support room on Matrix](https://img.shields.io/matrix/mash-playbook:devture.com.svg?label=%23mash-playbook%3Adevture.com&logo=matrix&style=for-the-badge&server_fqdn=matrix.devture.com&fetchMode=summary)](https://matrixrooms.info/room/mash-playbook:devture.com) [![donate](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/mother-of-all-self-hosting/donate)

# MASH Playbook: Your Ultimate Self-Hosting Toolkit

**Tired of juggling multiple playbooks and complex setups?** The MASH (Mother-of-All-Self-Hosting) Ansible playbook simplifies self-hosting by deploying and managing a wide array of services in Docker containers on your own server.

**[Visit the original repository](https://github.com/mother-of-all-self-hosting/mash-playbook) for the latest updates and information.**

## Key Features

*   **Simplified Self-Hosting:** Manage numerous services with a single, unified Ansible playbook.
*   **Containerized Deployments:** Utilize Docker containers for consistent and up-to-date service installations across multiple Linux distributions and CPU architectures.
*   **Extensive Service Support:** Easily deploy a growing list of [supported services](docs/supported-services.md).
*   **Automated Installation & Updates:** Leverage Ansible for seamless installation, configuration, and upgrades.
*   **Easy to Try New Services:** Quickly experiment with various services through simple configuration changes.
*   **Shared Services Management:** Centralized management of shared services like databases and reverse proxies.
*   **Simplified Backups:** Streamlined backup procedures due to all services residing under a single base data path.

## Supported Services

Discover the full list of supported services in the [supported services documentation](docs/supported-services.md).

## Getting Started

Follow the instructions in the [installation guide](docs/README.md) to configure and install services on your server.

## Staying Updated

Refer to the [changelog](CHANGELOG.md) to stay informed about updates and any potential backward-incompatible changes.

## Support and Community

*   **Matrix:** Join the community in the [#mash-playbook:devture.com](https://matrixrooms.info/room/mash-playbook:devture.com) Matrix room. You can join using a public server like `matrix.org` or self-host Matrix using the [matrix-docker-ansible-deploy](https://github.com/spantaleev/matrix-docker-ansible-deploy) playbook.
*   **GitHub Issues:** Report issues and contribute to the project via [GitHub issues](https://github.com/mother-of-all-self-hosting/mash-playbook/issues).

## Why MASH?

MASH consolidates the functionality of multiple service-specific playbooks (like those for Matrix, Nextcloud, Gitea, etc.) into a single, easy-to-manage solution. This approach streamlines deployment, reduces duplication of effort, and simplifies the process of self-hosting a wide variety of services. With MASH, you can easily create your ideal self-hosted stack, backed by a reliable and well-maintained Ansible playbook.

## What's in a name?

The name "MASH" reflects the playbook's purpose: to mix and match various services, providing a comprehensive toolkit for self-hosting, much like mashing ingredients in the brewing process to create a satisfying final product.