[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate Your Infrastructure with Ease

**Ansible is a powerful, open-source automation tool that simplifies IT tasks like configuration management and application deployment.**  [View the original repository on GitHub](https://github.com/ansible/ansible).

## Key Features

*   **Agentless Architecture:** Manages systems over SSH, eliminating the need for agents and reducing complexity.
*   **Configuration Management:** Automates the configuration of servers, applications, and services.
*   **Application Deployment:**  Streamlines application deployment across various environments.
*   **Cloud Provisioning:**  Simplifies the provisioning of cloud infrastructure.
*   **Orchestration:** Orchestrates multi-tier deployments and complex workflows.
*   **Ad-hoc Task Execution:** Executes commands and tasks on remote systems without pre-defined playbooks.
*   **Network Automation:** Automates network device configuration and management.
*   **Human-Readable Automation:**  Uses YAML to describe infrastructure in a simple and easily understood format.
*   **Parallel Execution:** Manages machines quickly and in parallel, saving time.
*   **Security Focused:** Prioritizes security and auditability.

## Getting Started with Ansible

### Installation

Install Ansible with `pip` or your preferred package manager. See the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions.

### Development Branch

Power users and developers can use the `devel` branch, which contains the latest features and fixes.  Be aware that this branch may have breaking changes.

## Community and Communication

Engage with the Ansible community for support and collaboration:

*   **[Ansible Forum](https://forum.ansible.com/c/help/6):** Ask questions, get help, and share your expertise.  Use tags like `ansible`, `ansible-core`, and `playbook` to find relevant discussions.
*   **[Social Spaces](https://forum.ansible.com/c/chat/4):** Connect with other Ansible enthusiasts.
*   **[News & Announcements](https://forum.ansible.com/c/news/5):** Stay up-to-date on project-wide announcements.
*   **[Bullhorn Newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn):** Receive release announcements and important updates.
*   **[Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html)**:  Find more ways to connect.

## Contributing to Ansible

Contribute to the project by:

*   Reviewing the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   Exploring [Community Information](https://docs.ansible.com/ansible/devel/community) for various contribution methods, including bug reports and code submissions.
*   Submitting pull requests to the `devel` branch.
*   Communicating before larger changes to coordinate efforts.

## Coding Guidelines

Review the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) for coding guidelines, particularly:

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`:  The active development branch.
*   `stable-2.X`:  Stable release branches.
*   Create branches from `devel` for pull requests and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup).
*   [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) for branch details.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details upcoming features and how to influence the roadmap.

## Authors and License

*   Created by [Michael DeHaan](https://github.com/mpdehaan) with contributions from a large community.
*   Sponsored by [Red Hat, Inc.](https://www.redhat.com)
*   Licensed under the GNU General Public License v3.0 or later (see [COPYING](COPYING)).