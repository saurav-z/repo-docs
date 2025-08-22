[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Simple IT Automation for Everyone

**Ansible is a powerful, open-source automation engine that simplifies IT tasks, saving you time and reducing errors.** This is the source repository for Ansible, learn more on the [Ansible website](https://ansible.com/) and in the [Ansible documentation](https://docs.ansible.com/ansible/latest/).

**[View the original Ansible repository on GitHub](https://github.com/ansible/ansible)**

## Key Features

*   **Agentless Architecture:** No agents to install or maintain, using SSH for secure connections.
*   **Configuration Management:** Automate the configuration of systems and applications.
*   **Application Deployment:** Streamline the deployment of applications across your infrastructure.
*   **Cloud Provisioning:** Easily manage cloud resources and infrastructure as code.
*   **Orchestration:** Coordinate complex multi-tier deployments and workflows.
*   **Ad-Hoc Task Execution:** Run commands and scripts on your infrastructure instantly.
*   **Network Automation:** Automate network device configuration and management.
*   **Idempotent Operations:** Ensures that tasks are only executed if necessary, avoiding unnecessary changes.
*   **Human-Readable Language:** Infrastructure described using YAML, making it easy to understand and maintain.

## Core Design Principles

*   **Ease of Use:** Simple setup and minimal learning curve.
*   **Parallel Execution:** Manage machines quickly and in parallel.
*   **Security First:** Focus on security and easy auditability.
*   **Extensible:** Module development is possible in any dynamic language.
*   **No Bootstrapping:** Manage new remote machines instantly.
*   **Non-Root Usage:** Usable as a non-root user.

## Getting Started

### Installation

Install Ansible using `pip` or your preferred package manager.  Refer to the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions on various platforms.

### Development Branch

For power users and developers, the `devel` branch provides the latest features and fixes, but may contain breaking changes. Engage with the Ansible community if you choose to use the `devel` branch.

## Communication and Community

Join the Ansible community to ask questions, get help, and share your knowledge.

*   **[Ansible Forum](https://forum.ansible.com/c/help/6):** Find help, share knowledge, and interact with the community.
*   **[Social Spaces](https://forum.ansible.com/c/chat/4):** Connect with fellow Ansible enthusiasts.
*   **[News & Announcements](https://forum.ansible.com/c/news/5):** Stay informed about project-wide announcements.
*   **[Bullhorn Newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn):** Get release announcements and important updates.
*   **[Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html):** Learn about other ways to connect with the Ansible community.

## Contributing

Contribute to the Ansible project and help improve the community.

*   Review the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   Explore [Community Information](https://docs.ansible.com/ansible/devel/community) for contribution details, including bug reports and code submissions.
*   Submit pull requests to the `devel` branch.
*   Discuss larger changes beforehand to coordinate efforts.

## Coding Guidelines

Adhere to our [Coding Guidelines](https://docs.ansible.com/ansible/devel/dev_guide/) in the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/).  Specifically, review:

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`: Active development branch.
*   `stable-2.X`: Stable release branches.
*   Create branches based on `devel` for pull requests and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup).
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for more information.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details the future development plans and how to provide feedback.

## Authors and License

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and has contributions from thousands of users.  [Ansible](https://www.ansible.com) is sponsored by [Red Hat, Inc.](https://www.redhat.com).

Licensed under the GNU General Public License v3.0 or later. See [COPYING](COPYING) for the full license text.