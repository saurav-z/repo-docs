[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate Your Infrastructure with Ease

**Ansible is a powerful, open-source automation engine that simplifies IT tasks and streamlines your infrastructure management.**  This README provides an overview of the core features, usage, and contribution guidelines for Ansible. For the original source code, see the [Ansible GitHub Repository](https://github.com/ansible/ansible).

## Key Features of Ansible:

*   **Configuration Management:** Automate the configuration of servers and applications.
*   **Application Deployment:** Deploy applications consistently across your infrastructure.
*   **Cloud Provisioning:** Provision resources in the cloud with ease.
*   **Ad-Hoc Task Execution:** Execute commands and tasks on remote machines quickly.
*   **Network Automation:** Automate the configuration and management of network devices.
*   **Orchestration:** Coordinate multi-node operations for complex deployments and updates.
*   **Agentless Architecture:** No agents to install on managed nodes, leveraging SSH for simplicity.
*   **Human-Readable Automation:** Uses YAML-based playbooks for easy-to-understand infrastructure-as-code.
*   **Parallel Execution:** Manages machines quickly and efficiently in parallel.

## Getting Started with Ansible

You can install Ansible using `pip` or a package manager.  Refer to the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions on your platform.

Power users and developers can run the `devel` branch for the latest features and fixes, but be aware of potential breaking changes.  Consider getting involved in the Ansible community if you intend to use the `devel` branch.

## Community and Support

*   **Forum:** [Ansible Forum](https://forum.ansible.com/c/help/6) - Ask questions, get help, and connect with the Ansible community.
*   **Communication:** [Ansible Community Communication](https://docs.ansible.com/ansible/devel/community/communication.html) - Find mailing lists, chat, and other ways to connect.
*   **Newsletter:** [Ansible Bullhorn Newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn) - Stay up-to-date with announcements and changes.

## Contributing to Ansible

We welcome contributions!  Please review the following resources:

*   **Contributor's Guide:**  [./.github/CONTRIBUTING.md]
*   **Community Information:** [Community Information](https://docs.ansible.com/ansible/devel/community)
*   **Submit a Pull Request:** Submit code updates to the `devel` branch.
*   **Coding Guidelines:** [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) - Review the coding guidelines, particularly for developing modules.

## Branch Information

*   **`devel`:**  Active development branch.
*   **`stable-2.X`:** Stable release branches.
*   **Dev Environment:** Set up a development environment based on `devel` for pull requests (see [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup)).
*   **Release Information:** [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html)

## Roadmap

*   The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details planned features and provides a way to influence the project's direction.

## Authors and License

*   **Created by:** [Michael DeHaan](https://github.com/mpdehaan) and a community of over 5000 contributors.
*   **Sponsored by:** [Red Hat, Inc.](https://www.redhat.com)
*   **License:** GNU General Public License v3.0 or later (see [COPYING](COPYING) for details).