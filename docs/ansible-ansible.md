[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Radically Simple IT Automation

Ansible is a powerful and easy-to-use IT automation platform that simplifies configuration management, application deployment, and orchestration across your infrastructure.  Get started today at the [official Ansible repository](https://github.com/ansible/ansible).

## Key Features

*   **Agentless Architecture:**  Operates without requiring agents on managed nodes, using SSH for secure communication.
*   **Configuration Management:** Automate the configuration of systems and applications, ensuring consistency and compliance.
*   **Application Deployment:** Streamline the deployment of applications across various environments.
*   **Cloud Provisioning:**  Provision infrastructure on cloud platforms like AWS, Azure, and Google Cloud.
*   **Orchestration:**  Orchestrate multi-tier deployments and complex workflows.
*   **Idempotency:**  Ensures tasks are executed only when necessary, preventing unintended changes.
*   **Human-Readable:** Uses a simple, YAML-based language for playbooks, making automation accessible to everyone.
*   **Extensible:** Supports module development in any dynamic language, providing flexibility and customization.
*   **Network Automation:** Automate network device configuration and management.
*   **Security Focused:** Designed with security in mind, promoting secure and auditable automation practices.

## Core Design Principles

*   **Simple Setup & Learning:**  Minimal learning curve and an easy setup process.
*   **Parallel Execution:** Manages machines quickly and in parallel.
*   **Agentless:** Leverages existing SSH daemon.
*   **Human-Friendly Infrastructure Description:** Infrastructure described in a language that is both machine and human friendly.
*   **Security Focused:** Emphasis on security and easy auditability.
*   **Instant Management:** Manage new remote machines instantly.
*   **Language Agnostic Modules:** Allow module development in any dynamic language.
*   **Non-Root Usage:** Usable as non-root.
*   **Ease of Use:** The easiest IT automation system to use.

## Getting Started with Ansible

Install Ansible using `pip` or your system's package manager. Detailed installation instructions are available in the [Ansible Installation Guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

## Community and Communication

Engage with the Ansible community for support, collaboration, and to stay informed.

*   **Ansible Forum:**  Ask questions, get help, and share your knowledge: [Ansible Forum](https://forum.ansible.com/c/help/6)
*   **Social Spaces:** Connect with fellow enthusiasts: [Ansible Social Spaces](https://forum.ansible.com/c/chat/4)
*   **News & Announcements:** Stay updated on project news and announcements: [Ansible News & Announcements](https://forum.ansible.com/c/news/5)
*   **The Bullhorn Newsletter:** Get release announcements and important changes: [The Bullhorn](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn)
*   **Community Communication:**  Find additional communication channels: [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html)

## Contributing to Ansible

Contribute to the Ansible project and help shape the future of IT automation.

*   **Contributor's Guide:** Explore the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   **Community Information:** Learn about various contribution methods: [Community Information](https://docs.ansible.com/ansible/devel/community)
*   **Submit a Pull Request:**  Propose code updates through pull requests to the `devel` branch.

## Coding Guidelines

Adhere to the Ansible coding guidelines for module development.

*   **Developer Guide:** Review the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/).
*   **Module Development Checklist:** Review [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   **Best Practices:** Review [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

Understand the different Ansible branches and their purposes.

*   `devel`: The actively developed branch for the latest features and fixes.
*   `stable-2.X`:  Branches for stable releases.
*   **Branch Setup:** Create a branch based on `devel` for pull requests and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup).
*   **Release and Maintenance:** Information about active branches: [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html)

## Roadmap

Stay informed about the future direction of Ansible.

*   **Ansible Roadmap:**  View the planned features and influence the roadmap: [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/)

## Authors and License

*   **Created by:** [Michael DeHaan](https://github.com/mpdehaan)
*   **Contributors:**  Over 5000 users have contributed (and growing!).
*   **Sponsored by:** [Red Hat, Inc.](https://www.redhat.com)
*   **License:** GNU General Public License v3.0 or later ([COPYING](COPYING))