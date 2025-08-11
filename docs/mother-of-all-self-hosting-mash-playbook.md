[![Support room on Matrix](https://img.shields.io/matrix/mash-playbook:devture.com.svg?label=%23mash-playbook%3Adevture.com&logo=matrix&style=for-the-badge&server_fqdn=matrix.devture.com&fetchMode=summary)](https://matrixrooms.info/room/mash-playbook:devture.com) [![donate](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/mother-of-all-self-hosting/donate)

# MASH: The Ultimate Self-Hosting Ansible Playbook

**Simplify your self-hosting journey with MASH, a comprehensive Ansible playbook that empowers you to deploy and manage a wide range of self-hosted services in Docker containers.**

[Visit the original repository on GitHub](https://github.com/mother-of-all-self-hosting/mash-playbook)

## Key Features

*   **Simplified Self-Hosting:** Easily deploy and manage numerous self-hosted services using a single, unified Ansible playbook.
*   **Containerized Deployments:** Leverage Docker containers for consistent, predictable, and up-to-date service installations across various Linux distributions and CPU architectures.
*   **Extensive Service Support:** Deploy a [wide array of supported services](docs/supported-services.md), with ongoing expansion to include more free and open-source software (FOSS).
*   **Automated Management:** Utilize Ansible for streamlined installation, upgrades, and maintenance tasks, ensuring an efficient self-hosting experience.
*   **Unified Infrastructure:** Benefit from shared services like PostgreSQL and Traefik reverse proxy, simplifying configuration and management.
*   **Easy Service Integration:** Effortlessly incorporate new services with minimal configuration, enabling quick experimentation and customization.
*   **Simplified Backups:** Centralized management of services and data paths makes backups straightforward.

## Supported Services

Explore the full list of supported services [here](docs/supported-services.md).

## Getting Started

To install and configure services on your server, follow the instructions in the [README](docs/README.md) located in the `docs/` directory.

## Staying Up-to-Date

The MASH playbook is constantly evolving. Stay informed about changes and updates by reviewing the [CHANGELOG](CHANGELOG.md).

## Support and Community

*   **Matrix Chat:** Connect with the community in the [#mash-playbook:devture.com](https://matrixrooms.info/room/mash-playbook:devture.com) Matrix room. You can use a public server like `matrix.org` or self-host Matrix.
*   **GitHub Issues:** Report bugs, request features, and engage in discussions via the [GitHub issues](https://github.com/mother-of-all-self-hosting/mash-playbook/issues).

## Why MASH? A Unified Approach to Self-Hosting

MASH addresses the challenges of managing multiple, separate playbooks by consolidating service deployments. This unified approach simplifies your workflow, reduces duplication, and makes it easy to add new services. It also ensures shared services, like databases and reverse proxies, are maintained in one place. Whether you are new to self-hosting or a veteran, MASH provides a powerful toolkit to create a personalized, self-hosted infrastructure.

## MASH vs. Dedicated Playbooks

MASH replaces the need to juggle multiple playbooks, such as:

*   [Matrix](https://github.com/spantaleev/matrix-docker-ansible-deploy)
*   [Nextcloud](https://github.com/spantaleev/nextcloud-docker-ansible-deploy)
*   [Gitea](https://github.com/spantaleev/gitea-docker-ansible-deploy)
*   [Gitlab](https://github.com/spantaleev/gitlab-docker-ansible-deploy)
*   [Vaultwarden](https://github.com/spantaleev/vaultwarden-docker-ansible-deploy)
*   [PeerTube](https://github.com/spantaleev/peertube-docker-ansible-deploy)
*   And more...

The [Matrix playbook](https://github.com/spantaleev/matrix-docker-ansible-deploy) remains separate, as it contains a huge number of components.

## The MASH Name

"MASH" stands for Mother-of-All-Self-Hosting, and represents the playbook's goal to be your all-in-one solution for self-hosting, allowing you to mix and mash various software components to build the self-hosted setup of your dreams.