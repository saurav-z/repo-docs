[![Support room on Matrix](https://img.shields.io/matrix/mash-playbook:devture.com.svg?label=%23mash-playbook%3Adevture.com&logo=matrix&style=for-the-badge&server_fqdn=matrix.devture.com&fetchMode=summary)](https://matrixrooms.info/room/mash-playbook:devture.com) [![donate](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/mother-of-all-self-hosting/donate)

# MASH Playbook: Your Ultimate Self-Hosting Toolkit

**MASH** (Mother-of-All-Self-Hosting) is an Ansible playbook designed to simplify and streamline self-hosting a wide variety of services using Docker containers. 

**[Visit the original repository](https://github.com/mother-of-all-self-hosting/mash-playbook) for the latest updates and features.**

## Key Features:

*   **Simplified Self-Hosting:** Easily deploy and manage numerous services on your server using Docker containers.
*   **Wide Service Support:** Supports a growing list of [FOSS](https://en.wikipedia.org/wiki/Free_and_open-source_software) services, see the [full list here](docs/supported-services.md).
*   **Containerized Deployment:** Leverage Docker containers for a predictable, up-to-date, and consistent setup across various Linux distributions and CPU architectures.
*   **Automated Installation & Updates:**  Benefit from automated installation, upgrades, and maintenance tasks using Ansible.
*   **Simplified Management:**  Avoid juggling multiple Ansible playbooks by using a single, comprehensive solution.
*   **Shared Service Management:** Shared services like PostgreSQL are managed in one single place.
*   **Easy Backups:** Makes backups easy, because everything lives together (same base data path, same Postgres instance)
*   **Supports many services**, previously supported by other playbooks ([Matrix](https://github.com/spantaleev/matrix-docker-ansible-deploy), [Nextcloud](https://github.com/spantaleev/nextcloud-docker-ansible-deploy), [Gitea](https://github.com/spantaleev/gitea-docker-ansible-deploy), [Gitlab](https://github.com/spantaleev/gitlab-docker-ansible-deploy), [Vaultwarden](https://github.com/spantaleev/vaultwarden-docker-ansible-deploy), [PeerTube](https://github.com/spantaleev/peertube-docker-ansible-deploy))

## Getting Started

Follow the instructions in the [README in the docs/ directory](docs/README.md) to configure and install services on your server.  For more details on using Ansible, consult our [Ansible guide](docs/ansible.md).

## Staying Updated

Review the [CHANGELOG.md](CHANGELOG.md) to stay informed about changes and backward-incompatible updates to the playbook.

## Support and Community

*   **Matrix:** Join the community in the [#mash-playbook:devture.com](https://matrixrooms.info/room/mash-playbook:devture.com) Matrix room. You can use a public server like `matrix.org` or self-host your own using the [matrix-docker-ansible-deploy](https://github.com/spantaleev/matrix-docker-ansible-deploy) playbook.
*   **GitHub Issues:** Report issues and contribute via [mother-of-all-self-hosting/mash-playbook/issues](https://github.com/mother-of-all-self-hosting/mash-playbook/issues).