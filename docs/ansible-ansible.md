# Ansible: Automate IT with Radically Simple Automation

Ansible is a powerful and easy-to-use IT automation engine for configuration management, application deployment, and orchestration.  Learn more and explore the official repository on [GitHub](https://github.com/ansible/ansible).

[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

## Key Features

*   **Agentless Architecture:** Uses SSH for communication, eliminating the need for agents and open ports.
*   **Configuration Management:** Automates the configuration of systems and applications.
*   **Application Deployment:** Streamlines the deployment process with easy-to-use playbooks.
*   **Cloud Provisioning:**  Provisions infrastructure on various cloud platforms.
*   **Orchestration:**  Automates multi-tier application deployment and zero-downtime rolling updates.
*   **Ad-hoc Task Execution:** Executes commands and tasks on remote machines quickly.
*   **Network Automation:** Automates network device configuration and management.
*   **Human-Readable Infrastructure as Code:** Uses YAML to describe infrastructure, making it easy to read, write, and maintain.

## Design Principles

*   Simple setup process with a minimal learning curve.
*   Fast and parallel machine management.
*   Focus on security, auditability, and easy content review.
*   Instant management of new remote machines, without bootstrapping.
*   Module development in any dynamic language.
*   Usable as non-root.
*   The easiest IT automation system to use.

## Getting Started

Install Ansible using `pip` or your preferred package manager.  Detailed installation instructions are available in the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

Developers and power users can use the `devel` branch, which contains the latest features and fixes. Please be aware that this branch may be unstable and subject to breaking changes.

## Communication and Community

Join the Ansible community for support, collaboration, and to share your knowledge.

*   **Ansible Forum:** [Get Help](https://forum.ansible.com/c/help/6), [Social Spaces](https://forum.ansible.com/c/chat/4), [News & Announcements](https://forum.ansible.com/c/news/5), and [Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn).
*   **Community Information:** Explore additional communication channels at [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing

We welcome contributions!

*   Review the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   Explore [Community Information](https://docs.ansible.com/ansible/devel/community).
*   Submit code updates through a pull request to the `devel` branch.
*   Discuss significant changes before making them to avoid duplicate effort.

## Coding Guidelines

Adhere to our coding guidelines, detailed in the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/).  Specifically, review:

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`:  Active development branch.
*   `stable-2.X`: Stable release branches.
*   Create branches based on `devel` for pull requests, and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup).
*   See [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) for branch information.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details future plans and how to influence them.

## Authors and License

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan).

[Ansible](https://www.ansible.com) is sponsored by [Red Hat, Inc.](https://www.redhat.com)

Licensed under the GNU General Public License v3.0 or later.  See [COPYING](COPYING) for details.