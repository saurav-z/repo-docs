[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Simple IT Automation & Configuration Management

**Ansible is a powerful and easy-to-use automation platform for configuration management, application deployment, and orchestration.**  Get started with Ansible by visiting the [official GitHub repository](https://github.com/ansible/ansible).

## Key Features

*   **Agentless Architecture:**  Leverages SSH for secure and efficient communication, eliminating the need for agents and reducing complexity.
*   **Configuration Management:** Automates the configuration of servers, applications, and infrastructure with ease.
*   **Application Deployment:** Streamlines application deployment across multiple environments.
*   **Cloud Provisioning:**  Simplifies the provisioning of cloud resources.
*   **Orchestration:**  Coordinates complex multi-tier application deployments.
*   **Idempotent Operations:** Ensures consistent state by only making changes when necessary.
*   **Human-Readable YAML:** Defines infrastructure as code in a simple, understandable format.
*   **Parallel Execution:**  Manages machines quickly and efficiently through parallel processing.
*   **Extensible:** Supports module development in any dynamic language.

## How to Use Ansible

Ansible can be installed via `pip` or a package manager.  Refer to the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions.

Developers and power users can explore the `devel` branch, containing the latest features and fixes.  Note that this branch may be less stable and could include breaking changes.

## Community and Communication

*   **Forum:**  Join the [Ansible forum](https://forum.ansible.com/c/help/6) to ask questions, get help, and engage with the community.  Use tags like `ansible`, `ansible-core`, and `playbook` to filter posts.
*   **Social Spaces:** Connect with fellow enthusiasts in the [social spaces](https://forum.ansible.com/c/chat/4).
*   **News & Announcements:** Stay informed about project-wide announcements via the [News & Announcements](https://forum.ansible.com/c/news/5) section.
*   **Bullhorn Newsletter:** Receive release announcements and important updates by subscribing to the [Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn).
*   **Community Information:**  For more ways to connect with the Ansible community, see [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contribute to Ansible

*   **Contributor's Guide:** Review the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   **Community Information:**  Explore the [Community Information](https://docs.ansible.com/ansible/devel/community) for contribution guidelines, bug reporting, and code submission instructions.
*   **Pull Requests:** Submit code updates through pull requests to the `devel` branch.
*   **Early Communication:** Discuss significant changes beforehand to avoid duplicate efforts.

## Coding Guidelines

*   **Developer Guide:** Consult the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) for coding guidelines.
*   **Module Development:**  Pay close attention to [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html) and [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html).

## Branch Information

*   `devel`: Active development branch.
*   `stable-2.X`: Stable release branches.
*   Create a branch from `devel` for pull requests and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup).
*   Refer to the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for branch details.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details planned features and how to influence future development.

## Authors and License

Ansible was created by Michael DeHaan and is supported by a vibrant community with contributions from thousands of users.

[Ansible](https://www.ansible.com) is sponsored by [Red Hat, Inc.](https://www.redhat.com).

Licensed under the GNU General Public License v3.0 or later. See [COPYING](COPYING) for the full license text.