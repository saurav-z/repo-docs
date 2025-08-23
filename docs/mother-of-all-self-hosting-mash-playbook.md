[![Support room on Matrix](https://img.shields.io/matrix/mash-playbook:devture.com.svg?label=%23mash-playbook%3Adevture.com&logo=matrix&style=for-the-badge&server_fqdn=matrix.devture.com&fetchMode=summary)](https://matrixrooms.info/room/mash-playbook:devture.com) [![donate](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/mother-of-all-self-hosting/donate)

# MASH: Mother-of-All-Self-Hosting Ansible Playbook

**Simplify your self-hosting journey with MASH, an Ansible playbook that lets you effortlessly deploy and manage a wide range of services in Docker containers.**  ([View the original repository](https://github.com/mother-of-all-self-hosting/mash-playbook))

MASH (Mother-of-All-Self-Hosting) provides a comprehensive solution for self-hosting your favorite services, ensuring a consistent and reliable setup across various Linux distributions and CPU architectures. Utilizing the power of Ansible and Docker, MASH simplifies the complexities of self-hosting, allowing you to focus on using your services rather than managing them.

## Key Features

*   **Simplified Deployment:** Automates the installation and configuration of numerous services using Docker containers.
*   **Wide Service Support:** Includes a growing list of supported services (see [Supported Services](docs/supported-services.md)), with a focus on open-source software (FOSS).
*   **Consistent Environment:** Leverages Docker to create a predictable and up-to-date environment for all your services.
*   **Automated Management:** Uses Ansible to automate installation, upgrades, and maintenance tasks.
*   **Easy to Expand:**  Designed to easily integrate new services and features.
*   **Centralized Management:** Manage your entire self-hosted ecosystem from a single Ansible playbook, reducing complexity and potential conflicts.
*   **Simplified Backups:**  Offers a convenient, single location for backups by design.

## Supported Services

Explore the extensive list of supported services in the [Supported Services](docs/supported-services.md) documentation.

## Installation & Usage

To get started with MASH and deploy services on your server, follow the detailed instructions in the [installation guide](docs/README.md) located within the `docs/` directory.  The guide covers configuration and the steps needed to configure and run MASH.

## Staying Up-to-Date

The MASH playbook is continuously evolving.  Refer to the [CHANGELOG](CHANGELOG.md) for detailed information on updates, new features, and any backward-incompatible changes to ensure a smooth and informed experience.

## Get Support

*   **Matrix:** Join the community in the `#mash-playbook:devture.com` Matrix room for support and discussions. Connect via a public server like `matrix.org` or self-host your own Matrix instance using [matrix-docker-ansible-deploy](https://github.com/spantaleev/matrix-docker-ansible-deploy).
*   **GitHub Issues:** Report bugs, request features, or get help with your MASH setup by opening an issue on GitHub: [mother-of-all-self-hosting/mash-playbook/issues](https://github.com/mother-of-all-self-hosting/mash-playbook/issues)

## Project Goals

MASH aims to streamline self-hosting by:

*   Eliminating the need to manage multiple Ansible playbooks.
*   Making it easy to experiment with and deploy various services.
*   Providing a home for smaller, single-purpose applications that might not warrant their own playbook.
*   Ensuring a consistent and reliable self-hosting experience.
*   Simplifying shared services like databases and backups.

## Why "Mother-of-All-Self-Hosting"?

The name reflects our ambition to provide a central, all-encompassing solution for your self-hosting needs.  Like the mashing process in brewing, MASH combines various software components to create a robust and versatile self-hosted environment, empowering you to build the stack of your dreams.