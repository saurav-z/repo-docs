[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate Everything with Ease

Ansible is a powerful and simple IT automation engine that simplifies configuration management, application deployment, and orchestration.  Get started today and explore the official [Ansible repository](https://github.com/ansible/ansible)!

## Key Features

*   **Agentless Architecture:** Uses SSH for secure, agent-free operation, eliminating the need for extra software on managed nodes.
*   **Simplified Configuration Management:** Streamlines infrastructure as code with human-readable YAML playbooks.
*   **Multi-Node Orchestration:** Easily manage and orchestrate tasks across multiple servers and systems.
*   **Application Deployment:** Automate the deployment of applications with minimal downtime.
*   **Cloud Provisioning:** Provision resources on various cloud platforms with ease.
*   **Network Automation:** Automate network device configuration and management.
*   **Ad-hoc Task Execution:** Execute commands on remote hosts quickly and efficiently.
*   **Zero-Downtime Updates:**  Allows for complex changes like zero-downtime rolling updates.
*   **Extensible:** Supports module development in any dynamic language.

## Getting Started

### Installation

Install the latest stable version of Ansible using `pip` or your preferred package manager.  Refer to the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions.

### Development Branch

For access to the newest features and fixes, power users and developers can run the `devel` branch.  Note that this branch may have breaking changes.  It's recommended to engage with the Ansible community when using the `devel` branch.

## Community and Support

*   **Get Help:**  Find answers, share your knowledge, and interact with the community on the [Ansible forum](https://forum.ansible.com/c/help/6).  Filter posts using tags such as `#ansible`, `#ansible-core`, and `#playbook`.
*   **Social Spaces:** Connect with fellow enthusiasts in the [Social Spaces](https://forum.ansible.com/c/chat/4).
*   **News & Announcements:** Stay informed about project-wide announcements via the [News & Announcements](https://forum.ansible.com/c/news/5) section.
*   **Newsletter:** Subscribe to the [Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn) for release announcements and important updates.
*   **Communication:** For more ways to get in touch, see [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing

We welcome contributions!  Please review the following resources:

*   **Contributor's Guide:**  Start with the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   **Community Information:** Explore various ways to contribute to the project in the [Community Information](https://docs.ansible.com/ansible/devel/community) section.
*   **Code Submissions:** Submit code updates via pull requests to the `devel` branch.
*   **Large Changes:** Discuss larger changes beforehand to avoid duplicate efforts.

## Coding Guidelines

Follow the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) for coding guidelines. Key sections include:

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`:  The active development branch.
*   `stable-2.X`:  Stable release branches.

Create a branch based on `devel` and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) if you want to open a PR.
*   For more details, see the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page.

## Roadmap

See the [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) for details on upcoming features and how to influence the roadmap.

## Authors and License

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and is a project with over 5000 contributors.

Ansible is sponsored by [Red Hat, Inc.](https://www.redhat.com).

Licensed under the GNU General Public License v3.0 or later.  See [COPYING](COPYING) for the full license text.