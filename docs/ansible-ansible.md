[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Radically Simple IT Automation

Ansible is a powerful, open-source automation platform that simplifies IT tasks like configuration management, application deployment, and cloud provisioning.  For more information, visit the [Ansible website](https://ansible.com/).

Want to get started? Check out the [original repo](https://github.com/ansible/ansible).

## Key Features of Ansible:

*   **Agentless Architecture:** Operates over SSH, eliminating the need for agents on managed nodes.
*   **Configuration Management:** Automates system configuration and ensures consistency.
*   **Application Deployment:** Streamlines the deployment of applications across various environments.
*   **Cloud Provisioning:**  Manages cloud infrastructure and resources.
*   **Orchestration:** Coordinates multi-node tasks and workflows.
*   **Ad-Hoc Task Execution:** Executes commands and scripts on remote machines quickly.
*   **Network Automation:** Automates network device configuration and management.
*   **Human-Readable & Machine-Friendly:** Uses a simple, YAML-based language for describing infrastructure.
*   **Parallel Execution:** Manages machines quickly and in parallel.
*   **Secure and Auditable:**  Focuses on security and easy auditability of changes.
*   **Easy to Get Started:**  Has a minimal learning curve and a simple setup process.

## Design Principles:

*   Simple setup process with a minimal learning curve.
*   Manage machines quickly and in parallel.
*   Avoid custom-agents and additional open ports, be agentless by
    leveraging the existing SSH daemon.
*   Describe infrastructure in a language that is both machine and human
    friendly.
*   Focus on security and easy auditability/review/rewriting of content.
*   Manage new remote machines instantly, without bootstrapping any
    software.
*   Allow module development in any dynamic language, not just Python.
*   Be usable as non-root.
*   Be the easiest IT automation system to use, ever.

## Getting Started with Ansible

Install Ansible with `pip` or your preferred package manager. The [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) provides detailed instructions for various platforms.

Power users and developers can explore the `devel` branch, though it's more prone to breaking changes. Join the Ansible community if you wish to use it.

## Community & Communication

*   **Forums:** [Ansible Forum](https://forum.ansible.com/c/help/6) - Get help, share your knowledge, and connect with the community.
*   **Social Spaces:**  [Ansible Forum](https://forum.ansible.com/c/chat/4) - Meet and interact with fellow enthusiasts.
*   **News & Announcements:**  [Ansible Forum](https://forum.ansible.com/c/news/5) - Stay updated on project-wide announcements.
*   **Bullhorn Newsletter:** [Ansible Community Communication](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn) - Receive release announcements and important updates.

Find more ways to connect with the community at [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing to Ansible

*   Review the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   Explore [Community Information](https://docs.ansible.com/ansible/devel/community) for contribution details, including bug reports and code submissions.
*   Submit code updates through pull requests to the `devel` branch.
*   Discuss larger changes beforehand to coordinate efforts.

## Coding Guidelines

Consult the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) for coding guidelines, especially:

*   [Contributing Your Module](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Best Practices](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`: Active development branch.
*   `stable-2.X`: Stable release branches.
*   Create a branch from `devel` and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) if you are opening a PR.
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for information about active branches.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) provides information on planned features and how to influence future development.

## Authors

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and benefits from contributions from a large community.

Sponsored by [Red Hat, Inc.](https://www.redhat.com)

## License

GNU General Public License v3.0 or later.
See [COPYING](COPYING) for the full license text.