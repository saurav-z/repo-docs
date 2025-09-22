# Ansible: Automate IT Infrastructure with Radically Simple Automation

[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

Ansible is a powerful, open-source automation platform that simplifies IT tasks, from configuration management to application deployment.  [Visit the original Ansible repository on GitHub](https://github.com/ansible/ansible) for the source code.

## Key Features & Benefits

*   **Agentless Architecture:** Operates over SSH, eliminating the need for agents and reducing complexity.
*   **Simple Setup & Learning Curve:** Easy to install and get started with, minimizing the time to automation.
*   **Parallel Execution:** Manages machines quickly and efficiently in parallel, saving time.
*   **Human-Readable Automation:** Uses YAML-based playbooks that are easy to understand and maintain.
*   **Comprehensive Automation:** Supports configuration management, application deployment, cloud provisioning, ad-hoc task execution, and network automation.
*   **Security Focused:** Designed with security in mind, promoting auditability and reviewability.
*   **Extensible:** Allows module development in any dynamic language.
*   **Idempotent Operations:** Ensures that tasks are executed only if necessary, preventing unintended changes.
*   **Zero Downtime Updates:** Facilitates complex changes like rolling updates with load balancers.

## Getting Started with Ansible

### Installation

Install Ansible using `pip` or your preferred package manager. Detailed installation instructions are available in the [Ansible Installation Guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

### Development Branch

Power users and developers can use the `devel` branch, which contains the latest features and fixes. Note that breaking changes are more likely in the `devel` branch.

## Communication & Community

Connect with the Ansible community for support, discussions, and contributions.

*   **Ansible Forum:** Ask questions, get help, and share your knowledge:
    *   [Help](https://forum.ansible.com/c/help/6)
    *   [Social Spaces](https://forum.ansible.com/c/chat/4)
    *   [News & Announcements](https://forum.ansible.com/c/news/5)
*   **Ansible Documentation:** Find more ways to get in touch here:  [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).
*   **Bullhorn Newsletter:** Stay up-to-date with releases and important changes:  [the-bullhorn](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn)

## Contributing to Ansible

Contribute to the Ansible project by following these guidelines:

*   **Contributor's Guide:** Refer to the [Contributor's Guide](./.github/CONTRIBUTING.md) for detailed contribution instructions.
*   **Community Information:**  Learn about different ways to contribute to the project in [Community Information](https://docs.ansible.com/ansible/devel/community).
*   **Submit Pull Requests:**  Propose code updates through pull requests to the `devel` branch.
*   **Discuss Larger Changes:** Discuss significant changes beforehand to coordinate efforts.

## Coding Guidelines

Adhere to these coding guidelines to ensure consistency and quality:

*   **Developer Guide:**  Review the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) for comprehensive coding guidelines.
*   **Module Development:**  Follow the [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html) and [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html) for best practices.

## Branch Information

Understand the different branches and their purposes:

*   `devel`: Active development branch.
*   `stable-2.X`: Stable release branches.
*   Create a branch based on `devel` and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) if you want to open a PR.
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for information about active branches.

## Roadmap

*   Refer to the [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) for information about planned features and releases.

## Authors & License

*   **Created by:** [Michael DeHaan](https://github.com/mpdehaan)
*   **Contributions from:** Over 5000 users (and growing)
*   **Sponsored by:** [Red Hat, Inc.](https://www.redhat.com)
*   **License:** GNU General Public License v3.0 or later (see [COPYING](COPYING))