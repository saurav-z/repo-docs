[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate Your Infrastructure with Ease

Ansible is a powerful, open-source automation engine that simplifies IT tasks like configuration management and application deployment.  For more information, see the [Ansible Website](https://ansible.com/).

**Looking for the core Ansible project?  You're in the right place!  This is the source code repository for Ansible, originally created by Michael DeHaan, and sponsored by Red Hat, Inc. [Visit the original repository on GitHub](https://github.com/ansible/ansible).**

## Key Features & Benefits

*   **Agentless Architecture:** Uses SSH for communication, eliminating the need for agents and reducing complexity.
*   **Configuration Management:** Automates the configuration of systems and applications.
*   **Application Deployment:** Simplifies the deployment of applications across multiple servers.
*   **Orchestration:** Enables the coordination of complex multi-tier deployments.
*   **Cloud Provisioning:** Automates the provisioning of infrastructure in the cloud.
*   **Simple and Human-Readable:** Uses YAML, making automation code easy to read, write, and maintain.
*   **Parallel Execution:** Manages machines quickly and in parallel, improving efficiency.
*   **Security Focused:** Prioritizes security with easy auditability and review of content.
*   **Extensible:** Allows module development in any dynamic language.

## Getting Started with Ansible

### Installation

Install Ansible using `pip` or your preferred package manager. Comprehensive installation instructions are available in the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

### Running the `devel` Branch

Power users and developers can run the `devel` branch, which has the latest features and fixes.  It is reasonably stable, but you are more likely to encounter breaking changes.  Get involved in the Ansible community if you want to run the `devel` branch.

## Community & Support

### Communication

*   **[Ansible Forum](https://forum.ansible.com/c/help/6):** Ask questions, get help, and share your Ansible knowledge.
*   **[Social Spaces](https://forum.ansible.com/c/chat/4):** Connect with fellow Ansible users.
*   **[News & Announcements](https://forum.ansible.com/c/news/5):** Stay updated on project-wide announcements.
*   **[Bullhorn Newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn):** Get release announcements and important changes.
*   For more ways to get in touch, see [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

### Contributing

Contribute to the Ansible project!

*   **[Contributor's Guide](./.github/CONTRIBUTING.md)**
*   **[Community Information](https://docs.ansible.com/ansible/devel/community)** for all
    kinds of ways to contribute to and interact with the project,
    including how to submit bug reports and code to Ansible.
*   Submit a proposed code update through a pull request to the `devel` branch.
*   Talk to us before making larger changes
    to avoid duplicate efforts.

## Development Information

### Coding Guidelines

Review the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) for detailed coding guidelines, especially:

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

### Branch Information

*   `devel`: Active development branch.
*   `stable-2.X`: Stable release branches.
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for more information.

### Roadmap

Explore the [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) to learn about upcoming features and influence the project's direction.

## Authors & License

*   **Created by:** [Michael DeHaan](https://github.com/mpdehaan)
*   **Sponsored by:** [Red Hat, Inc.](https://www.redhat.com)
*   **License:** GNU General Public License v3.0 or later (see [COPYING](COPYING)).