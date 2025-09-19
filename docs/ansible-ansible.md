[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate IT with Ease

**Ansible is a powerful and simple IT automation engine that simplifies configuration management, application deployment, and more.** This project is the core of Ansible, available on [GitHub](https://github.com/ansible/ansible).

## Key Features

*   **Agentless Architecture:** Ansible uses SSH for secure and agentless connections to manage machines.
*   **Configuration Management:** Automate the configuration of systems and applications.
*   **Application Deployment:** Streamline application deployment across multiple environments.
*   **Cloud Provisioning:** Easily provision infrastructure on various cloud platforms.
*   **Orchestration:** Manage multi-tier deployments and complex tasks with ease.
*   **Ad-hoc Task Execution:** Execute commands and tasks on remote systems quickly.
*   **Network Automation:** Automate network device configuration and management.
*   **Human-Readable Automation:** Describe infrastructure in a simple, machine- and human-friendly language (YAML).

## Core Design Principles

*   **Simplicity:**  Easy to set up with a minimal learning curve.
*   **Parallelism:**  Manages machines quickly and concurrently.
*   **Security:** Prioritizes secure configurations and easy auditability.
*   **Idempotency:**  Ensures tasks are executed only when necessary.
*   **Extensibility:** Supports module development in various languages.

## Getting Started with Ansible

### Installation

Install Ansible using `pip` or your preferred package manager.  Refer to the [Ansible installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions.

### Development Version

For access to the latest features and fixes, you can run the `devel` branch.  Be aware that this branch may include breaking changes.  Consult the [Ansible community](https://docs.ansible.com/ansible/devel/community/index.html) for more information.

## Community and Contribution

### Communication

Join the Ansible community to ask questions, get help, and connect with other users:

*   **[Ansible Forum](https://forum.ansible.com/):** Ask questions, share knowledge.
    *   Explore topics using tags such as: [ansible](https://forum.ansible.com/tag/ansible), [ansible-core](https://forum.ansible.com/tag/ansible-core), [playbook](https://forum.ansible.com/tag/playbook).
*   **[Social Spaces](https://forum.ansible.com/c/chat/4):** Engage with fellow enthusiasts.
*   **[News & Announcements](https://forum.ansible.com/c/news/5):** Stay updated on project news.
*   **[Bullhorn Newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn):**  Receive release announcements.

### Contributing

Contribute to Ansible by:

*   Reviewing the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   Consulting the [Community Information](https://docs.ansible.com/ansible/devel/community) page.
*   Submitting pull requests to the `devel` branch.
*   Discussing larger changes beforehand to coordinate efforts.

### Coding Guidelines

*   Review the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) for coding guidelines.
*   Pay special attention to the sections on:
    *   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
    *   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`:  The active development branch.
*   `stable-2.X`:  Stable release branches.
*   Create a branch based on `devel` and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) to open a PR.
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for branch information.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details planned features and allows you to influence the direction of the project.

## Authors and License

*   Created by [Michael DeHaan](https://github.com/mpdehaan) and a growing community of contributors.
*   Sponsored by [Red Hat, Inc.](https://www.redhat.com)
*   Licensed under the GNU General Public License v3.0 or later (see [COPYING](COPYING)).