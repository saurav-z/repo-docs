[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Simple IT Automation for Everyone

Ansible is a powerful, open-source automation engine that simplifies IT tasks like configuration management, application deployment, and cloud provisioning.  [Explore the original repository](https://github.com/ansible/ansible) for more information and contribute to the project.

## Key Features

*   **Agentless Architecture:** Uses SSH for management, eliminating the need for agents and open ports.
*   **Simplified Setup:**  Easy to install and configure with a minimal learning curve.
*   **Parallel Execution:** Manages machines quickly and concurrently.
*   **Human-Readable Language:** Uses YAML to describe infrastructure, making it easy to understand and maintain.
*   **Comprehensive Automation:** Supports configuration management, application deployment, cloud provisioning, and more.
*   **Security Focused:** Designed with security in mind, promoting auditability and review.
*   **Multi-Node Orchestration:** Simplifies complex tasks like zero-downtime rolling updates.
*   **Extensible:** Allows module development in any dynamic language.

## Getting Started with Ansible

### Installation

You can install a released version of Ansible using `pip` or your preferred package manager.  Refer to the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions.

### Development Branch

Power users and developers can leverage the `devel` branch, which includes the latest features and fixes. Note that this branch may have breaking changes.

## Communication and Community

Connect with the Ansible community for support, discussions, and collaboration.

*   **Forums:** [Ansible Forum](https://forum.ansible.com/c/help/6) for questions, help, and knowledge sharing.
*   **Social Spaces:** [Social Spaces](https://forum.ansible.com/c/chat/4) to meet and interact with the community.
*   **News & Announcements:** [News & Announcements](https://forum.ansible.com/c/news/5) to stay updated.
*   **Newsletter:** [Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn) for release announcements and updates.
*   **Other Communication:** [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html)

## Contributing to Ansible

Help improve Ansible by contributing to the project.

*   **Contributor's Guide:** Review the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   **Community Information:** Explore [Community Information](https://docs.ansible.com/ansible/devel/community) for details on contributing.
*   **Submit Code:** Submit pull requests to the `devel` branch.
*   **Discuss Changes:** Discuss significant changes beforehand to avoid duplicated effort.

## Coding Guidelines

Follow the coding guidelines to maintain the quality and consistency of the project.  Refer to the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/).

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`: The development branch, containing the latest features.
*   `stable-2.X`: Stable release branches.
*   Create a branch based on `devel` and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) if you want to open a PR.
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for information about active branches.

## Roadmap

Consult the [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) for planned features and influence the project's direction.

## Authors

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and has contributions from over 5000 users.

[Ansible](https://www.ansible.com) is sponsored by [Red Hat, Inc.](https://www.redhat.com)

## License

GNU General Public License v3.0 or later

See [COPYING](COPYING) for the full license text.