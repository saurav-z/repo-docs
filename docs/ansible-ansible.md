[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate IT with Radically Simple Automation

Ansible is a powerful, open-source automation platform designed to simplify IT tasks, including configuration management, application deployment, and cloud provisioning. Visit the [Ansible GitHub Repository](https://github.com/ansible/ansible) for more information.

## Key Features

*   **Agentless Architecture:** Uses SSH, eliminating the need for agents and open ports.
*   **Simple Setup & Learning Curve:** Designed for easy setup and a minimal learning curve.
*   **Parallel Execution:** Manages machines quickly and in parallel for efficiency.
*   **Human-Readable Automation:** Infrastructure described in a clear, easy-to-understand language.
*   **Security-Focused:** Emphasizes security and easy auditability.
*   **Modules for Any Language:**  Develop modules in any dynamic language, not just Python.
*   **Zero-Downtime Deployments:** Simplifies complex tasks like rolling updates with load balancers.
*   **Network Automation:** Enables automation across network devices.
*   **Multi-Node Orchestration:** Orchestrates complex tasks across multiple machines.

## Getting Started

### Installation

Install Ansible using `pip` or your preferred package manager.  Refer to the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions.

### Development Branch

Power users and developers can utilize the `devel` branch for the latest features and fixes. Note that this branch is more likely to include breaking changes. Engage with the Ansible community for support when using the `devel` branch.

## Communication & Community

*   **Ansible Forum:**  A hub for asking questions, getting help, and connecting with the Ansible community ([Forum Link](https://forum.ansible.com/c/help/6)).
    *   Find help using tags such as:  `ansible`, `ansible-core`, or `playbook`.
*   **Social Spaces:** Interact with other enthusiasts in the [Social Spaces forum](https://forum.ansible.com/c/chat/4).
*   **News & Announcements:** Stay updated on project-wide news and events on the [News & Announcements forum](https://forum.ansible.com/c/news/5).
*   **Bullhorn Newsletter:** Receive release announcements and important updates ([Newsletter Link](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn)).

For additional ways to get involved and communicate with the community, see [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing

Contribute to the project by:

*   Reviewing the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   Consulting [Community Information](https://docs.ansible.com/ansible/devel/community) for various ways to contribute.
*   Submitting pull requests to the `devel` branch.
*   Discussing significant changes with the community beforehand.

## Coding Guidelines

Find our Coding Guidelines in the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/). Review these documents, especially:

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`: Actively developed release branch.
*   `stable-2.X`: Stable release branches.
*   When opening a PR, create a branch based on `devel` and set up your development environment ([environment setup](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup)).

Refer to the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for details about active branches.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) provides details of planned features and how to influence future development.

## Authors & License

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and has contributions from a large community. Ansible is sponsored by [Red Hat, Inc.](https://www.redhat.com).

**License:** GNU General Public License v3.0 or later. See [COPYING](COPYING) for the full license text.