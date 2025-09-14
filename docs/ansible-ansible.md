[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Simple IT Automation for Everyone

**Ansible** is a powerful, open-source automation platform that simplifies IT tasks like configuration management, application deployment, and cloud provisioning. ([See the original repo](https://github.com/ansible/ansible))

## Key Features of Ansible:

*   **Agentless Architecture:** Operates over SSH, eliminating the need for agents on managed nodes and simplifying setup.
*   **Configuration Management:** Automates the configuration of systems and applications.
*   **Application Deployment:** Streamlines the deployment of applications across various environments.
*   **Cloud Provisioning:**  Easily provisions infrastructure on cloud platforms.
*   **Orchestration:**  Orchestrates complex multi-tier application deployments.
*   **Ad-hoc Task Execution:** Executes commands and tasks on remote systems without writing playbooks.
*   **Network Automation:** Automates the configuration and management of network devices.
*   **Human-Readable Automation:** Uses a simple, YAML-based language for defining automation tasks.
*   **Parallel Execution:** Manages machines quickly and in parallel.
*   **Security Focus:** Prioritizes security and easy auditability.

## Getting Started with Ansible

### Installation

Install Ansible using `pip` or a package manager.  Refer to the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions for your platform.

### Devel Branch

For the latest features and fixes, consider the `devel` branch.  Be aware of potential breaking changes.  Join the Ansible community for support and discussions.

## Communication & Community

*   **Forum:** Get help, share knowledge, and interact with the community on the [Ansible forum](https://forum.ansible.com/c/help/6).
*   **Social Spaces:** Connect with fellow enthusiasts in [social spaces](https://forum.ansible.com/c/chat/4).
*   **News & Announcements:** Stay updated on project-wide announcements and events in the [News & Announcements](https://forum.ansible.com/c/news/5) section.
*   **Bullhorn Newsletter:** Receive release announcements and important changes via the [Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn).
*   **More ways to get in touch:** [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing to Ansible

Contribute to the project by:

*   Reviewing the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   Exploring the [Community Information](https://docs.ansible.com/ansible/devel/community).
*   Submitting code updates via pull requests to the `devel` branch.
*   Discussing larger changes beforehand to avoid duplication.

## Coding Guidelines & Development

*   Review the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) for detailed coding guidelines.
*   Specifically, see [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html) and [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html).

## Branch Information

*   `devel`: Active development branch.
*   `stable-2.X`: Stable release branches.
*   Create a branch from `devel` for pull requests and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup).
*   Review the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page.

## Roadmap

Find details on planned features and influence the direction of Ansible on the [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/).

## Authors & License

*   Created by [Michael DeHaan](https://github.com/mpdehaan).
*   Thanks to contributions from over 5000 users.
*   Sponsored by [Red Hat, Inc.](https://www.redhat.com)
*   License: GNU General Public License v3.0 or later.  See [COPYING](COPYING).