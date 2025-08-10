[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate IT with Simplicity and Power

Ansible is a powerful, open-source IT automation engine that simplifies configuration management, application deployment, and cloud provisioning, making complex tasks like zero-downtime updates easy to manage.  ([Original Repo](https://github.com/ansible/ansible))

## Key Features

*   **Agentless Architecture:** Operates over SSH, eliminating the need for agents on managed nodes, simplifying setup and security.
*   **Configuration Management:**  Automates the configuration of systems and applications, ensuring consistency and reducing manual effort.
*   **Application Deployment:** Streamlines the deployment process, ensuring applications are delivered reliably and efficiently.
*   **Cloud Provisioning:**  Automates the creation and management of cloud infrastructure, supporting various providers.
*   **Orchestration:**  Orchestrates multi-tier deployments, allowing for complex automation workflows across multiple systems.
*   **Ad-Hoc Task Execution:** Executes commands and tasks on remote systems quickly and efficiently.
*   **Network Automation:** Automates network device configuration and management.
*   **Human-Readable Automation:** Uses YAML to describe automation tasks, making them easy to read, write, and understand.
*   **Parallel Execution:** Manages machines quickly and in parallel.

## Getting Started

Install Ansible using `pip` or your preferred package manager.  Detailed installation instructions are available in the [Ansible Installation Guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

Power users and developers can run the `devel` branch, which has the latest features and fixes, directly. Although it is reasonably stable, you are more likely to encounter breaking changes when running the `devel` branch. We recommend getting involved in the Ansible community if you want to run the `devel` branch.

## Community and Communication

Connect with the Ansible community for support, collaboration, and announcements:

*   **Forum:**  [Ansible Forum](https://forum.ansible.com/c/help/6) - Ask questions, share knowledge, and get help from the community.
    *   Use tags like `ansible`, `ansible-core`, and `playbook` to filter and subscribe to relevant discussions.
*   **Social Spaces:** [Ansible Social Spaces](https://forum.ansible.com/c/chat/4) - Engage with fellow enthusiasts.
*   **News & Announcements:** [Ansible News & Announcements](https://forum.ansible.com/c/news/5) - Stay informed about project updates and events.
*   **Newsletter:** [The Bullhorn Newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn) - Get release announcements and important updates.
*   **Other Channels:** For more ways to get in touch, see [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing

Contribute to the Ansible project and help improve IT automation:

*   **Contributor's Guide:**  Review the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   **Community Information:** Explore [Community Information](https://docs.ansible.com/ansible/devel/community) for contribution guidelines and community interactions.
*   **Submit Pull Requests:** Propose code updates to the `devel` branch.
*   **Discuss Changes:** Discuss larger changes before implementation to avoid duplicate efforts.

## Coding Guidelines

Follow the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) for coding standards and best practices. Specifically review:

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

Understand the different Ansible branches:

*   `devel`: Actively developed release.
*   `stable-2.X`: Stable releases.

Create a branch based on `devel` and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) if you want to open a PR.

See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for information about active branches.

## Roadmap

*   The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details planned features and how to influence the roadmap.

## Authors and License

*   **Created by:** [Michael DeHaan](https://github.com/mpdehaan) with contributions from thousands of users.
*   **Sponsored by:** [Red Hat, Inc.](https://www.redhat.com)
*   **License:** GNU General Public License v3.0 or later (see [COPYING](COPYING)).