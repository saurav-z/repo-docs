[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate IT Infrastructure and Simplify Configuration Management

Ansible is a powerful and open-source IT automation engine that simplifies configuration management, application deployment, and cloud provisioning. This README provides an overview of Ansible's features, design principles, and how to get started.  For more detailed information, visit the [official Ansible repository](https://github.com/ansible/ansible).

## Key Features

*   **Agentless Architecture:** Operates over SSH, eliminating the need for agents on managed nodes.
*   **Simple Setup:** Easy to install and use with a minimal learning curve.
*   **Parallel Execution:** Manages machines quickly and efficiently in parallel.
*   **Human-Readable Language:** Uses YAML to describe infrastructure, making it easy to understand and maintain.
*   **Modules for Everything:**  Offers a comprehensive set of modules for tasks like configuration management, application deployment, and cloud provisioning.
*   **Security Focused:** Designed with security and auditability in mind.
*   **Idempotent:**  Ensures that tasks are only executed if necessary, avoiding unwanted changes.
*   **Extensible:** Supports module development in any dynamic language.

## Design Principles

Ansible is built on several core principles:

*   **Simplicity:** Easy to learn and use, with a focus on a straightforward setup process.
*   **Speed:**  Designed to manage machines quickly and efficiently, utilizing parallel execution.
*   **Agentless:** Leverages SSH for communication, avoiding the need for agents and additional open ports.
*   **Human-Friendly Infrastructure as Code:** Uses a clear, declarative language (YAML) for describing infrastructure.
*   **Security-First:** Prioritizes security and easy auditability of all configurations.
*   **Instant Remote Machine Management:** Enables immediate management of new remote machines without bootstrapping software.
*   **Language Agnostic Modules:** Allows module development in any dynamic language, not just Python.
*   **Usable Without Root Access:** Provides the flexibility to be used as a non-root user.
*   **Ease of Use:** Strives to be the most user-friendly IT automation system available.

## Getting Started with Ansible

You can install Ansible using `pip` or a package manager.

For detailed installation instructions, please see the [Ansible Installation Guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

## Community and Communication

The Ansible community is active and supportive. Here's how to connect:

*   **Forum:**  Ask questions, get help, and share your knowledge on the [Ansible Forum](https://forum.ansible.com/c/help/6).
*   **Social Spaces:** Interact with fellow enthusiasts in the [Social Spaces](https://forum.ansible.com/c/chat/4).
*   **News & Announcements:** Stay up-to-date on project-wide announcements on the [News & Announcements](https://forum.ansible.com/c/news/5) section.
*   **Bullhorn Newsletter:** Receive release announcements and important changes via the [Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn).
*   **Other Communication Channels:** Find additional ways to get in touch with the community at [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing to Ansible

We welcome contributions from the community! Here's how you can get involved:

*   **Contributor's Guide:** Review the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   **Community Information:** Explore various ways to contribute to and interact with the project in the [Community Information](https://docs.ansible.com/ansible/devel/community) section. This includes submitting bug reports and code.
*   **Pull Requests:** Submit your code updates through a pull request to the `devel` branch.
*   **Collaboration:**  Talk to us before making larger changes to avoid duplicate efforts.

## Coding Guidelines

Review the following resources for coding guidelines:

*   [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/)
*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`:  The active development branch.
*   `stable-2.X`: Stable release branches.
*   Create branches based on `devel` and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) to open a PR.
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) for information about active branches.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details planned features and how to influence the roadmap.

## Authors and License

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and has contributions from over 5000 users. Thanks everyone!

Ansible is sponsored by [Red Hat, Inc.](https://www.redhat.com)

**License:**  GNU General Public License v3.0 or later. See [COPYING](COPYING) for the full license text.