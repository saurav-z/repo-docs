[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate IT with Radical Simplicity

Ansible is a powerful, open-source IT automation engine that simplifies configuration management, application deployment, cloud provisioning, and more.

**[View the original repository on GitHub](https://github.com/ansible/ansible)**

## Key Features & Benefits

*   **Agentless Architecture:** Manages systems using SSH, eliminating the need for agents and reducing complexity.
*   **Configuration Management:** Automates the configuration of systems, ensuring consistency and reducing manual errors.
*   **Application Deployment:** Streamlines the deployment of applications across your infrastructure, making it faster and more reliable.
*   **Cloud Provisioning:** Easily provisions and manages cloud resources, enabling infrastructure-as-code practices.
*   **Orchestration:** Orchestrates multi-tier deployments and complex tasks across multiple servers.
*   **Human-Readable YAML:** Uses YAML to describe infrastructure in a format that's both machine- and human-friendly.
*   **Security Focused:** Designed with security in mind, offering easy auditability and review capabilities.
*   **Extensible:** Allows module development in any dynamic language.
*   **Multi-Node Orchestration:** Simplifies complex changes like zero-downtime rolling updates.

## Core Design Principles

*   Easy setup with minimal learning curve.
*   Fast and parallel machine management.
*   Agentless architecture leveraging SSH.
*   Human-readable infrastructure description.
*   Focus on security and auditability.
*   Instant management of new remote machines.
*   Module development in any dynamic language.
*   Usable as non-root.

## Getting Started

### Installation

Install Ansible using `pip` or a package manager. For detailed instructions, see the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

### Development Branch

For developers and power users, the `devel` branch offers the latest features and fixes. Note that this branch may include breaking changes.  Consider joining the Ansible community if you use this branch.

## Community and Support

### Getting Help

*   **Ansible Forum:** Ask questions, get help, and interact with the community.
    *   [Help](https://forum.ansible.com/c/help/6): Get assistance and share your Ansible knowledge.
    *   [Social Spaces](https://forum.ansible.com/c/chat/4): Connect with other enthusiasts.
    *   [News & Announcements](https://forum.ansible.com/c/news/5): Stay up-to-date with project announcements.
*   **Ansible Community Communication:** See [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html) for more ways to connect.

### Newsletter

*   [Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn): Receive release announcements and important updates.

## Contributing

We welcome contributions!

*   [Contributor's Guide](./.github/CONTRIBUTING.md)
*   [Community Information](https://docs.ansible.com/ansible/devel/community): Learn how to contribute, report bugs, and submit code.
*   Submit pull requests to the `devel` branch.

## Coding Guidelines

Review the coding guidelines in the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/), especially:

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`: Active development branch.
*   `stable-2.X`: Stable release branches.
*   Use `devel` to create branches when creating a PR.
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for details about active branches.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details future plans and how to influence the roadmap.

## Authors and License

*   Created by [Michael DeHaan](https://github.com/mpdehaan) and the contributions of over 5000 users (and growing).
*   Sponsored by [Red Hat, Inc.](https://www.redhat.com)
*   Licensed under the GNU General Public License v3.0 or later. See [COPYING](COPYING) for the full license text.