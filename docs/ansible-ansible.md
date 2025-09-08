[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate Everything with Radically Simple IT Automation

Ansible is a powerful, open-source IT automation engine that simplifies configuration management, application deployment, cloud provisioning, and more.  [Visit the Ansible GitHub Repository](https://github.com/ansible/ansible) to learn more.

## Key Features

*   **Agentless Architecture:** Operates over SSH, eliminating the need for agents and reducing complexity.
*   **Configuration Management:** Automates the configuration of systems, ensuring consistency and reducing errors.
*   **Application Deployment:** Streamlines the deployment of applications across multiple environments.
*   **Cloud Provisioning:** Manages cloud resources, allowing for infrastructure as code.
*   **Orchestration:** Coordinates tasks across multiple nodes for complex operations like rolling updates.
*   **Human-Readable Language:** Uses YAML to describe infrastructure in a simple, understandable format.
*   **Security Focused:** Prioritizes security through auditability, reviewability, and ease of rewriting.

## Design Principles

Ansible is built on the following core principles:

*   **Simple Setup & Learning Curve:**  Easy to get started with minimal effort.
*   **Parallel Execution:** Manages machines quickly and efficiently in parallel.
*   **Agentless:** Leverages SSH for secure, agentless communication.
*   **Human-Friendly Language:** Uses a declarative language for infrastructure definitions.
*   **Security First:**  Prioritizes security and easy auditing.
*   **Instant Management:**  Manages new remote machines without requiring bootstrapping.
*   **Multi-Language Module Development:** Supports module development in any dynamic language.
*   **Non-Root Usability:**  Operates as a non-root user.
*   **Ease of Use:** Designed to be the easiest IT automation system available.

## Getting Started

You can install Ansible using `pip` or your system's package manager.  Refer to the [Ansible installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions.

Power users and developers can explore the `devel` branch, which contains the latest features and fixes.  However, be aware that the `devel` branch may have breaking changes.

## Community & Communication

Connect with the Ansible community for support, discussions, and contributions:

*   **Ansible Forum:**  [Get Help](https://forum.ansible.com/c/help/6), [Social Spaces](https://forum.ansible.com/c/chat/4), [News & Announcements](https://forum.ansible.com/c/news/5).
*   **Mailing Lists:**  Join the [Ansible mailing lists](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information) for announcements and discussions.
*   **Bullhorn Newsletter:** Subscribe for release announcements and important updates.

For comprehensive communication details, see [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing

Contribute to Ansible and help shape its future:

*   **Contributor's Guide:**  Explore the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   **Community Information:**  Learn about contributions and interactions in the [Community Information](https://docs.ansible.com/ansible/devel/community).
*   **Submit Pull Requests:**  Submit code updates via pull requests to the `devel` branch.

## Coding Guidelines

Adhere to the following coding guidelines for developing modules:

*   Review the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/).
*   Focus on [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html) and [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html).

## Branch Information

*   `devel`:  The actively developed branch.
*   `stable-2.X`:  Stable release branches.
*   Create a branch based on `devel` and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) if you want to open a PR.
*   Review the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for information about active branches.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details what is planned and how to influence the roadmap.

## Authors and License

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and has contributions from a large community.

Ansible is sponsored by [Red Hat, Inc.](https://www.redhat.com).

Licensed under the GNU General Public License v3.0 or later. See [COPYING](COPYING) for the full license text.