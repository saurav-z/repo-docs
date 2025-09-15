[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate Everything with Ease

Ansible is a powerful, open-source IT automation engine that simplifies configuration management, application deployment, and orchestration.  Learn more and contribute on the [original repository](https://github.com/ansible/ansible).

## Key Features

*   **Agentless Architecture:**  Uses SSH for remote execution, eliminating the need for agents and simplifying setup.
*   **Simple and Readable:**  Employs a human-readable, YAML-based language to describe infrastructure and automation tasks.
*   **Configuration Management:** Automate the configuration of systems and applications.
*   **Application Deployment:**  Streamline application deployment across multiple environments.
*   **Orchestration:** Orchestrate complex, multi-tier deployments and rolling updates.
*   **Cloud Provisioning:** Provision infrastructure on various cloud providers.
*   **Idempotent Operations:** Ensures that tasks are only executed if necessary, preventing unwanted changes.
*   **Extensible:** Supports module development in any dynamic language.
*   **Security-Focused:** Designed with security in mind, making it easy to audit and review automation code.
*   **Parallel Execution:**  Manages machines quickly and in parallel.
*   **Network Automation:**  Automates network device configuration and management.

## Getting Started with Ansible

### Installation

You can install Ansible using `pip` or a package manager. Consult the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions.

###  Communication & Support

*   **Community Forum:**  Ask questions, get help, and interact with the Ansible community on the [Ansible forum](https://forum.ansible.com/c/help/6).
*   **Social Spaces:** Engage with fellow enthusiasts in the [Social Spaces](https://forum.ansible.com/c/chat/4).
*   **News & Announcements:** Stay updated on project-wide announcements in the [News & Announcements](https://forum.ansible.com/c/news/5).
*   **Mailing Lists:** Subscribe to the [Ansible mailing lists](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information) for important updates.
*   **Bullhorn newsletter:** Subscribe to the [Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn) for release announcements and important changes.
*   **IRC Chat:** Find support and chat with other users on [IRC Chat](https://docs.ansible.com/ansible/devel/community/communication.html).

For more ways to get in touch, see [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing to Ansible

We welcome contributions from the community!

*   **Contributor's Guide:** Review the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   **Community Information:** Explore the [Community Information](https://docs.ansible.com/ansible/devel/community) for various ways to contribute.
*   **Submit a Pull Request:** Propose code updates through a pull request to the `devel` branch.
*   **Discuss Larger Changes:** Contact the team to discuss larger changes to avoid duplicate efforts.

## Coding Guidelines

Adhere to the coding guidelines detailed in the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/).  Specifically, review:

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`:  Active development branch.
*   `stable-2.X`: Stable release branches.
*   Create a branch based on `devel` and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) if you want to open a PR.
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for information about active branches.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details planned features and how to influence the roadmap.

## Authors

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and has contributions from over 5000 users.

## License

GNU General Public License v3.0 or later.  See [COPYING](COPYING) for the full license text.