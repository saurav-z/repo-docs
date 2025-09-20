[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate IT Tasks and Simplify Infrastructure Management

Ansible is a powerful and easy-to-use IT automation engine that streamlines configuration management, application deployment, and cloud provisioning. ([Original Repository](https://github.com/ansible/ansible))

## Key Features & Benefits

*   **Agentless Architecture:** Operates over SSH, eliminating the need for agents and reducing overhead.
*   **Configuration Management:** Automates system configuration, ensuring consistency and reducing manual errors.
*   **Application Deployment:** Simplifies the deployment of applications across multiple servers.
*   **Cloud Provisioning:** Provisions infrastructure on various cloud platforms.
*   **Orchestration:** Coordinates complex multi-tier deployments and updates.
*   **Ad-hoc Task Execution:** Executes commands and scripts on remote machines quickly.
*   **Network Automation:** Automates network device configuration and management.
*   **Human-Readable Infrastructure Definition:** Uses YAML to describe infrastructure, making it easy to understand and maintain.
*   **Parallel Execution:** Manages machines quickly and in parallel.
*   **Security Focused:** Emphasizes security and easy auditability.

## Getting Started with Ansible

### Installation

You can install Ansible using `pip` or your system's package manager. For detailed installation instructions, refer to the [Ansible Installation Guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

### Development Branch

Power users and developers can use the `devel` branch to explore the newest features and fixes, while understanding that it may have breaking changes.

## Community & Support

*   **Ansible Forum:** Get help, ask questions, and engage with the Ansible community via the [Ansible Forum](https://forum.ansible.com/).
    *   Filter posts by tags like [ansible](https://forum.ansible.com/tag/ansible), [ansible-core](https://forum.ansible.com/tag/ansible-core), and [playbook](https://forum.ansible.com/tag/playbook).
*   **Social Spaces:** Interact with fellow enthusiasts in the [Social Spaces](https://forum.ansible.com/c/chat/4).
*   **News & Announcements:** Stay updated on project announcements and events via the [News & Announcements](https://forum.ansible.com/c/news/5) section.
*   **Bullhorn Newsletter:** Receive release announcements and important updates by subscribing to the [Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn).

For additional ways to connect with the community, please see [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing

We welcome contributions from the community! Please review the following resources:

*   **Contributor's Guide:** Learn how to contribute to the project by reviewing the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   **Community Information:** Explore various ways to contribute and interact with the project, including submitting bug reports and code, in the [Community Information](https://docs.ansible.com/ansible/devel/community).
*   **Pull Requests:** Submit proposed code updates to the `devel` branch via pull requests.
*   **Large Changes:** Discuss large changes beforehand to avoid duplication of effort.

## Coding Guidelines

Adhering to the coding guidelines will facilitate consistent and maintainable code.  Refer to the following:

*   **Developer Guide:**  Access our Coding Guidelines in the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/).
*   **Module Development:** Review guidelines such as [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html) and [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html).

## Branch Information

*   `devel`: Corresponds to the release under active development.
*   `stable-2.X`: Represents stable releases.
*   Create a branch off of `devel` for your development and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) to submit a PR.
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for information about active branches.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) provides insight into future plans, and describes how you can influence the roadmap.

## Authors

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and is supported by a large community with over 5000 contributors.

## License

Ansible is licensed under the GNU General Public License v3.0 or later. See [COPYING](COPYING) for the full license text.