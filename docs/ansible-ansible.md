[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate Your Infrastructure with Ease

Ansible is a powerful, open-source automation engine that simplifies IT tasks, from configuration management to application deployment, making infrastructure automation accessible to everyone. ([View the original repository](https://github.com/ansible/ansible))

## Key Features:

*   **Agentless Architecture:** Manages systems over SSH, eliminating the need for agents and simplifying setup.
*   **Configuration Management:** Automates system configuration, ensuring consistency and reducing manual errors.
*   **Application Deployment:** Streamlines application deployment across multiple servers with ease.
*   **Cloud Provisioning:** Simplifies cloud infrastructure management, supporting various cloud providers.
*   **Orchestration:** Orchestrates complex multi-tier deployments and updates with rolling updates.
*   **Human-Readable Playbooks:** Uses YAML to define infrastructure as code, making automation easy to understand and maintain.
*   **Extensible Modules:** Provides a rich library of modules and supports custom module development in any language.
*   **Security Focused:** Prioritizes security with easy auditability and review of automation content.
*   **Parallel Execution:** Executes tasks quickly and in parallel across your entire infrastructure.
*   **Idempotent Operations:** Ensures tasks are executed only when necessary, preventing unintended changes.

## Getting Started

### Installation

You can install a released version of Ansible using `pip` or your system's package manager.  Consult the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions.

### Devel Branch

For power users and developers, the `devel` branch offers the latest features and fixes.  Be aware that the `devel` branch may contain breaking changes. Consider getting involved with the Ansible community before using it.

## Community and Communication

Join the Ansible community to ask questions, get help, and interact with other enthusiasts.

*   **[Ansible Forum](https://forum.ansible.com/c/help/6):** Find help, share your knowledge, and connect with other users.
*   **[Social Spaces](https://forum.ansible.com/c/chat/4):** Meet and interact with fellow enthusiasts.
*   **[News & Announcements](https://forum.ansible.com/c/news/5):** Stay informed about project-wide announcements.
*   **[Bullhorn Newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn):** Receive release announcements and important updates.
*   For more ways to get in touch, see [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing

Contribute to Ansible and help shape the future of IT automation!

*   **[Contributor's Guide](./.github/CONTRIBUTING.md):** Learn how to contribute to the project.
*   **[Community Information](https://docs.ansible.com/ansible/devel/community):** Discover ways to interact with and contribute to the project, including submitting bug reports and code.
*   Submit a proposed code update through a pull request to the `devel` branch.
*   Discuss larger changes with the community beforehand to avoid duplicate efforts.

## Coding Guidelines

*   **[Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/):** Review our detailed coding guidelines.

    *   **[Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)**
    *   **[Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)**

## Branch Information

*   `devel`: Active development branch.
*   `stable-2.X`: Stable release branches.
*   Create a branch based on `devel` for pull requests and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup).
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for branch information.

## Roadmap

*   The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details planned features and allows you to influence the roadmap.

## Authors and License

*   Created by [Michael DeHaan](https://github.com/mpdehaan).
*   Contributions from over 5000 users.
*   Sponsored by [Red Hat, Inc.](https://www.redhat.com)
*   **License:** GNU General Public License v3.0 or later ([COPYING](COPYING))