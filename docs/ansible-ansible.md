[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Radically Simple IT Automation

Ansible is an open-source IT automation engine that simplifies configuration management, application deployment, and cloud provisioning.  [Explore the original Ansible repository on GitHub](https://github.com/ansible/ansible).

## Key Features

*   **Agentless Architecture:** Operates over SSH, eliminating the need for agents on managed nodes.
*   **Simple Setup & Learning Curve:** Easy to install and use, with a minimal learning curve.
*   **Parallel Execution:** Manages machines quickly and in parallel, significantly speeding up automation tasks.
*   **Human-Readable Automation:** Uses a declarative language (YAML) to describe infrastructure, making it easy to read, write, and audit.
*   **Modules for Extensibility:**  Supports module development in any dynamic language, not just Python.
*   **Security Focused:** Designed with security in mind, allowing for easy auditability and review.
*   **Idempotent Operations:**  Ensures that tasks are performed only when necessary, preventing unintended changes.
*   **Zero Downtime Deployments:** Facilitates complex changes, like zero-downtime rolling updates.
*   **Network Automation:** Ansible extends its capabilities to network devices, enabling automated configuration and management.

## Core Functionality

Ansible excels in several key areas:

*   **Configuration Management:** Automates the setup and maintenance of servers, applications, and services.
*   **Application Deployment:** Streamlines the deployment of applications across multiple environments.
*   **Cloud Provisioning:**  Automates the creation and management of cloud resources.
*   **Orchestration:**  Coordinates tasks across multiple systems, ensuring they are executed in the correct order.
*   **Ad-Hoc Task Execution:** Executes commands on remote machines without writing playbooks.

## Getting Started

### Installation

Install Ansible using `pip` or your system's package manager.  See the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for details.

### Community & Support

*   **Ansible Forum:**  Ask questions, get help, and interact with the community via the [Ansible Forum](https://forum.ansible.com/c/help/6).
*   **Community Information:**  Learn about contributing, reporting bugs, and more on the [Community Information](https://docs.ansible.com/ansible/devel/community) page.
*   **Communication:** Find other ways to get in touch via the [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html) page.

## Contributing

We welcome contributions!  Please review the following resources:

*   **Contributor's Guide:**  [Contributor's Guide](./.github/CONTRIBUTING.md).
*   **Coding Guidelines:**  [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) and review [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html) and [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html).
*   **Submit a Pull Request:**  Propose code updates to the `devel` branch.

## Branch Information

*   `devel`:  Active development branch.
*   `stable-2.X`: Stable release branches.
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for more information.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details planned features and how to influence the project's direction.

## Authors & License

*   Created by Michael DeHaan and thousands of contributors.
*   Sponsored by Red Hat, Inc.
*   Licensed under the GNU General Public License v3.0 or later (see [COPYING](COPYING)).