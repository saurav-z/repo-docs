[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate Your Infrastructure with Ease

Ansible is a powerful and easy-to-use IT automation engine that simplifies configuration management, application deployment, and more. [View the original repository on GitHub](https://github.com/ansible/ansible).

## Key Features of Ansible

*   **Agentless Architecture:** Operates over SSH, eliminating the need for agents on managed nodes.
*   **Configuration Management:** Automates the configuration of servers and other IT infrastructure.
*   **Application Deployment:** Streamlines the deployment of applications across your environment.
*   **Cloud Provisioning:** Automates the creation and management of cloud resources.
*   **Orchestration:** Coordinates multi-node tasks and complex workflows.
*   **Ad-hoc Task Execution:** Executes commands and scripts on remote systems.
*   **Network Automation:** Automates the configuration and management of network devices.
*   **Human-Readable Automation:** Uses a simple, YAML-based language for playbooks.
*   **Security Focused:** Designed with security, auditability, and ease of review in mind.
*   **Parallel Execution:** Manages machines quickly and in parallel for faster results.

## Getting Started with Ansible

### Installation

You can install Ansible using `pip` or a package manager.  Consult the [Ansible Installation Guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions on supported platforms.

### Running the Development Branch

For access to the latest features and fixes, you can run the `devel` branch, but be aware that it may contain breaking changes. The Ansible community is a great resource for support with the `devel` branch.

## Communication and Community

### Get Help

*   **Ansible Forum:** Ask questions and get assistance from the Ansible community at [Ansible Forum](https://forum.ansible.com/c/help/6).
    *   Utilize tags to find relevant information, such as:
        *   [ansible](https://forum.ansible.com/tag/ansible)
        *   [ansible-core](https://forum.ansible.com/tag/ansible-core)
        *   [playbook](https://forum.ansible.com/tag/playbook)
*   **Social Spaces:** Connect with fellow enthusiasts via the [Social Spaces](https://forum.ansible.com/c/chat/4).
*   **News & Announcements:** Stay informed about project-wide announcements and events at the [News & Announcements](https://forum.ansible.com/c/news/5).
*   **Bullhorn Newsletter:** Receive release announcements and important updates by subscribing to the [Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn).

### More Information

For additional ways to connect with the Ansible community, see [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing to Ansible

### How to Contribute

*   Review the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   Explore [Community Information](https://docs.ansible.com/ansible/devel/community) for detailed contribution guidelines, including bug reporting and code submission.
*   Submit code changes via pull requests to the `devel` branch.
*   Discuss substantial changes with the community beforehand to avoid duplicated efforts.

### Coding Guidelines

*   Find detailed information on coding guidelines in the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/).
    *   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
    *   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`: The active development branch.
*   `stable-2.X`: Branches for stable releases.
*   For pull requests, base your branch on `devel` and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup).
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) for information about active branches.

## Roadmap

Review the [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) for current and planned features, and for how to influence the roadmap.

## Authors and License

### Authors

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and has contributions from thousands of users.

### Sponsorship

[Ansible](https://www.ansible.com) is sponsored by [Red Hat, Inc.](https://www.redhat.com).

### License

GNU General Public License v3.0 or later.  See [COPYING](COPYING) for the complete license.