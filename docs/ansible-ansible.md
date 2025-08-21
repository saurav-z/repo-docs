[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate IT with Radically Simple Automation

Ansible is a powerful, open-source IT automation engine that simplifies configuration management, application deployment, cloud provisioning, and more.  [See the original repository](https://github.com/ansible/ansible).

## Key Features

*   **Agentless Architecture:**  Uses SSH for secure and efficient communication, eliminating the need for agents on managed nodes.
*   **Simple Setup & Learning Curve:** Designed for ease of use, making automation accessible to a wider audience.
*   **Parallel Execution:** Manages machines quickly and efficiently in parallel, saving time and resources.
*   **Human-Readable Language:**  Uses YAML to describe infrastructure as code, making automation tasks easy to understand and maintain.
*   **Security-Focused:**  Prioritizes security through design, making auditability and review straightforward.
*   **Modules in Any Language:** Enables module development in any dynamic language, providing flexibility and extensibility.
*   **Comprehensive Automation:** Automates configuration management, application deployment, cloud provisioning, ad-hoc task execution, network automation, and multi-node orchestration.
*   **Zero-Downtime Updates:** Simplifies complex tasks like zero-downtime rolling updates with load balancers.

## Getting Started with Ansible

You can easily install Ansible using `pip` or your preferred package manager. For detailed installation instructions, refer to the [Ansible Installation Guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

Power users and developers can also run the `devel` branch to access the latest features and fixes, though it may contain breaking changes.  The Ansible community is a great resource for assistance with the `devel` branch.

## Communication and Community

Engage with the Ansible community to ask questions, find solutions, and contribute your expertise.

*   **Ansible Forum:** [Get Help](https://forum.ansible.com/c/help/6), [Social Spaces](https://forum.ansible.com/c/chat/4), and [News & Announcements](https://forum.ansible.com/c/news/5)
*   **Bullhorn Newsletter:** Stay updated on release announcements and important changes: [The Bullhorn](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn).
*   **Community Communication:**  Explore additional ways to connect with the Ansible community at [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing to Ansible

Contribute to the project and make a difference!

*   **Contributor's Guide:**  Review the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   **Community Information:** Access resources for contributing and interacting with the project, including bug reports and code contributions in the [Community Information](https://docs.ansible.com/ansible/devel/community).
*   **Pull Requests:** Submit code updates via pull requests to the `devel` branch.
*   **Discuss Changes:**  Consult with the team before undertaking larger changes to avoid duplicate efforts.

## Coding Guidelines

Follow the established coding guidelines to ensure code quality and consistency.

*   **Developer Guide:** Consult the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) for coding conventions.
*   **Module Development:** Review these important resources for module development:
    *   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
    *   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

Understand the different branches for development and stable releases.

*   `devel`:  The branch for ongoing development.
*   `stable-2.X`:  Branches for stable releases.
*   Create a branch based on `devel` for your contributions and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup).
*   Review the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page to learn about active branches.

## Roadmap

*   The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) provides insights into future releases and allows you to influence the roadmap.

## Authors and License

*   **Created by:** [Michael DeHaan](https://github.com/mpdehaan) and a community of over 5000 contributors.
*   **Sponsored by:** [Red Hat, Inc.](https://www.redhat.com)
*   **License:** GNU General Public License v3.0 or later ([COPYING](COPYING))