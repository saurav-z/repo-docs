[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate IT with Radical Simplicity

Ansible is a powerful, open-source IT automation engine that simplifies configuration management, application deployment, cloud provisioning, and more.  Visit the [Ansible GitHub Repository](https://github.com/ansible/ansible) for the source code.

## Key Features

*   **Agentless Architecture:** No agents or additional open ports needed, utilizing existing SSH.
*   **Simple Setup:**  Easy to learn and implement with a minimal learning curve.
*   **Parallel Execution:** Manage machines quickly and efficiently in parallel.
*   **Human-Readable Language:** Uses YAML to describe infrastructure, making it easy to read and understand.
*   **Security-Focused:** Designed with security, auditability, and review in mind.
*   **Idempotent Operations:** Ensures that tasks are executed only when necessary, preventing unwanted changes.
*   **Extensible Modules:**  Allows module development in any dynamic language, not just Python.
*   **Cloud Agnostic:** Supports various cloud platforms including AWS, Azure, and Google Cloud Platform.
*   **Network Automation:** Automate network device configuration and management.

## Getting Started

### Installation

Install Ansible using `pip` or your preferred package manager.  Refer to the detailed [Installation Guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for installation instructions on various platforms.

### Running the `devel` Branch

Power users and developers can run the `devel` branch to access the latest features and fixes. However, be aware that the `devel` branch is more prone to breaking changes. Engage with the Ansible community for support and collaboration if you choose to use this branch.

##  Community and Support

*   **[Ansible Forum](https://forum.ansible.com/):** Get help, ask questions, and connect with the Ansible community.
    *   **Help:** Find solutions and share your Ansible knowledge.
    *   **Social Spaces:** Interact with fellow Ansible enthusiasts.
    *   **News & Announcements:** Stay informed about project-wide updates.
    *   **Bullhorn Newsletter:** Receive release announcements and important information.
*   **[Communication](https://docs.ansible.com/ansible/devel/community/communication.html):** Explore more ways to connect with the Ansible community.

## Contributing

Contribute to Ansible and help shape the future of automation!

*   **[Contributor's Guide](./.github/CONTRIBUTING.md):** Learn how to contribute.
*   **[Community Information](https://docs.ansible.com/ansible/devel/community):** Discover different ways to engage with the project.
*   **Submit a PR:** Propose code updates to the `devel` branch.
*   **Discussion:** Discuss larger changes to avoid duplicate work.

### Coding Guidelines

*   **[Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/):** Review our coding guidelines.
    *   **Contributing your module to Ansible**: Checklist
    *   **Conventions, tips, and pitfalls:** Best Practices

## Branch Information

*   `devel`: Active development branch.
*   `stable-2.X`: Stable releases.
*   Create a branch based on `devel` for PRs and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) if you want to open a PR.
*   [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) details active branches.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details planned features and how to influence the roadmap.

## Authors and License

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and is licensed under the GNU General Public License v3.0 or later. See [COPYING](COPYING) for the full license text.  The project has received contributions from over 5000 users. Thanks everyone!

Ansible is sponsored by [Red Hat, Inc.](https://www.redhat.com)