[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Simple IT Automation for Everyone

Ansible is a powerful, open-source automation platform that simplifies IT tasks, making infrastructure management and application deployment easier than ever. [Learn more about Ansible on GitHub](https://github.com/ansible/ansible).

**Key Features:**

*   **Agentless Architecture:** No need to install agents on managed nodes, leveraging SSH for secure connections.
*   **Configuration Management:** Automate the configuration of systems, ensuring consistency and compliance.
*   **Application Deployment:** Streamline application deployment across various environments.
*   **Cloud Provisioning:** Manage and provision resources in the cloud with ease.
*   **Orchestration:** Coordinate complex multi-tier deployments and tasks.
*   **Human-Readable Automation:** Describe infrastructure in a simple, declarative language (YAML) that's easy to understand.
*   **Parallel Execution:** Manage machines quickly and in parallel, saving time and resources.
*   **Security Focused:** Emphasizes security, auditability, and easy review of content.
*   **Modular Design:** Allows module development in any dynamic language.
*   **Idempotent Operations:** Ensures that tasks are executed only when necessary, preventing unintended changes.
*   **Zero-Downtime Rolling Updates:** Facilitates complex changes, like zero-downtime rolling updates with load balancers.

## Getting Started

### Installation

Install a released version of Ansible using `pip` or your preferred package manager. Refer to the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions.

### Development Branch

Power users and developers can run the `devel` branch, which contains the latest features and fixes.  However, be aware that this branch may have breaking changes.

## Community & Support

*   **Forums:** [Ansible Forum](https://forum.ansible.com/): Ask questions, get help, and connect with the Ansible community.
*   **Communication:** [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html)
    *   **Mailing Lists:** Stay informed through the Ansible mailing lists.
    *   **Bullhorn Newsletter:** Get release announcements and important changes.

## Contributing

Contribute to the project and help make Ansible even better!

*   [Contributor's Guide](https://github.com/ansible/ansible/blob/devel/.github/CONTRIBUTING.md)
*   [Community Information](https://docs.ansible.com/ansible/devel/community)
*   Submit code updates via pull requests to the `devel` branch.

## Development Guidelines

*   [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/)
    *   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
    *   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`:  Active development branch for new features.
*   `stable-2.X`: Branches for stable releases.
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for details.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) outlines planned features and how to influence future development.

## Authors and License

*   Created by [Michael DeHaan](https://github.com/mpdehaan).
*   Thanks to over 5000 contributors!
*   Sponsored by [Red Hat, Inc.](https://www.redhat.com)
*   **License:** GNU General Public License v3.0 or later. See [COPYING](COPYING) for full details.