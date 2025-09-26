[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Simple IT Automation for Everyone

**Ansible** is a powerful, open-source IT automation engine that simplifies configuration management, application deployment, cloud provisioning, and more.  For more information, see the [Ansible website](https://ansible.com/)

[View the original repository on GitHub](https://github.com/ansible/ansible)

## Key Features

*   **Agentless Architecture:** Uses SSH for secure communication, eliminating the need for agents on managed nodes.
*   **Simple Setup & Learning Curve:** Designed for ease of use, with a minimal learning curve, making it accessible to all.
*   **Parallel Execution:** Manages machines quickly and in parallel, saving time and resources.
*   **Human-Readable Infrastructure as Code:** Describes infrastructure in a clear, machine-friendly, and human-readable language (YAML).
*   **Extensible & Flexible:** Allows module development in any dynamic language.
*   **Security Focused:** Designed with security best practices in mind, promoting easy auditability.
*   **Idempotent Operations:** Ensures that tasks are only executed if necessary, avoiding unintended changes.

## Getting Started

### Installation

Install Ansible using `pip` or your preferred package manager.  Detailed installation instructions can be found in the [Ansible Installation Guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

### Running the Development Branch

Power users and developers can run the `devel` branch, which has the latest features and fixes. Although it is reasonably stable, you are more likely to encounter breaking changes when running the `devel` branch. We recommend getting involved
in the Ansible community if you want to run the `devel` branch.

## Community & Support

Join the Ansible community to ask questions, get help, and connect with other users.

*   **Ansible Forum:**  [Get Help](https://forum.ansible.com/c/help/6), [Social Spaces](https://forum.ansible.com/c/chat/4), [News & Announcements](https://forum.ansible.com/c/news/5)
*   **Ansible Mailing Lists:** [Ansible Mailing Lists](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
*   **Bullhorn Newsletter:** Stay up-to-date with release announcements and important changes via the [Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn).

## Contributing

We welcome contributions! Please review the [Contributor's Guide](./.github/CONTRIBUTING.md) and [Community Information](https://docs.ansible.com/ansible/devel/community/) for guidelines on how to contribute.

*   Submit code updates via a pull request to the `devel` branch.
*   Discuss larger changes beforehand to coordinate efforts and avoid duplication.

### Coding Guidelines

Refer to the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) for our coding guidelines.  Specifically, review:

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`: Active development branch.
*   `stable-2.X`: Stable release branches.

See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for information about active branches.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details future plans and how to influence the roadmap.

## Authors & License

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and has contributions from over 5000 users.

Ansible is sponsored by [Red Hat, Inc.](https://www.redhat.com)

**License:** GNU General Public License v3.0 or later. See [COPYING](COPYING) for the full text.