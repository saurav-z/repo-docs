[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate Everything with Simple IT Automation

Ansible is a powerful and easy-to-use IT automation engine that simplifies configuration management, application deployment, cloud provisioning, and more. [View the original repo](https://github.com/ansible/ansible).

## Key Features

*   **Agentless Architecture:**  Operates over SSH, eliminating the need for agents and simplifying setup.
*   **Configuration Management:**  Automates the configuration of systems and applications.
*   **Application Deployment:**  Streamlines the deployment of applications across your infrastructure.
*   **Cloud Provisioning:**  Provisions cloud resources with ease.
*   **Orchestration:** Orchestrates multi-node operations for complex tasks.
*   **Idempotent Operations:**  Ensures that tasks are only executed if necessary, avoiding unwanted changes.
*   **Human-Readable Automation:** Uses YAML to describe automation tasks, making them easy to understand and maintain.
*   **Network Automation:** Automates network device configuration and management.

## Getting Started

### Installation

Install Ansible using `pip` or your preferred package manager.  Refer to the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions.

### Development Branch

The `devel` branch contains the latest features and fixes.  Be aware that it may include breaking changes.

## Communication and Community

Connect with the Ansible community for help, discussions, and announcements.

*   **Forum:**  [Ansible Forum](https://forum.ansible.com/): Ask questions, share knowledge, and interact with other users.
*   **Social Spaces:** [Ansible Social Spaces](https://forum.ansible.com/c/chat/4): Meet and interact with fellow enthusiasts.
*   **News & Announcements:** [Ansible News & Announcements](https://forum.ansible.com/c/news/5): Stay updated on project-wide announcements.
*   **Mailing Lists:** [Ansible Mailing Lists](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information):  Stay informed about the latest releases.
*   **Bullhorn Newsletter:** [The Bullhorn](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn): Receive release announcements and other important updates.

## Contributing

We welcome contributions!  Please review the following:

*   **Contributor's Guide:**  [Contributor's Guide](./.github/CONTRIBUTING.md).
*   **Community Information:** [Ansible Community Information](https://docs.ansible.com/ansible/devel/community): Learn about various ways to contribute.
*   **Submit a Pull Request:** Submit proposed code changes to the `devel` branch.

## Coding Guidelines

*   **Developer Guide:** [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/): Review our coding guidelines to ensure your contributions align with our standards.
    *   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
    *   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`:  Active development branch.
*   `stable-2.X`:  Stable release branches.
*   Create a branch based on `devel` and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) if you want to open a PR.
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for information about active branches.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details planned features and allows you to influence the project's direction.

## Authors

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and has contributions from thousands of users.

Sponsored by [Red Hat, Inc.](https://www.redhat.com)

## License

GNU General Public License v3.0 or later

See [COPYING](COPYING) for the full license text.