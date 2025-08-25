[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate IT Tasks with Ease

Ansible is a powerful and simple IT automation engine that streamlines configuration management, application deployment, and cloud provisioning.  For more information, please visit the [Ansible website](https://ansible.com/).  This is the open source project repository.

**[View the original Ansible repository on GitHub](https://github.com/ansible/ansible)**

## Key Features

*   **Agentless Architecture:** Leverages SSH for easy and secure connections, eliminating the need for agents and reducing complexity.
*   **Configuration Management:** Automates the setup and maintenance of your systems, ensuring consistency and compliance.
*   **Application Deployment:** Simplifies the process of deploying applications across your infrastructure, from simple apps to complex microservices.
*   **Cloud Provisioning:** Makes it easy to provision and manage resources on various cloud platforms.
*   **Orchestration:** Automates complex multi-tier deployments and rolling updates with zero downtime.
*   **Human-Readable Automation:** Uses a simple, YAML-based language that's easy to learn and understand.
*   **Modules in Any Language:** Allows the creation of modules in any dynamic language, increasing flexibility for developers.

## Key Benefits

*   **Easy to Use:**  Simple setup and a minimal learning curve gets you automating quickly.
*   **Efficient:** Manages machines quickly and in parallel, saving you time.
*   **Secure:** Focuses on security and easy auditability of your automation tasks.
*   **Flexible:**  Can be used as a non-root user, and supports many platforms.

## Getting Started

### Installation

Install a released version of Ansible using `pip` or your preferred package manager.  See the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for platform-specific instructions.

### Development Branch

Power users and developers can run the `devel` branch to experience the latest features and fixes. Be aware that this branch is less stable and may contain breaking changes.

## Communication & Community

Connect with the Ansible community to ask questions, get help, and contribute:

*   **Ansible Forum:** [Get Help](https://forum.ansible.com/c/help/6), [Social Spaces](https://forum.ansible.com/c/chat/4), and [News & Announcements](https://forum.ansible.com/c/news/5)
*   **Mailing Lists:** [Ansible Mailing Lists](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
*   **Newsletter:** [Bullhorn Newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn)

## Contributing

Contribute to Ansible and help improve the project:

*   **Contributor's Guide:** Review the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   **Community Information:**  Explore [Community Information](https://docs.ansible.com/ansible/devel/community) to understand ways to contribute.
*   **Pull Requests:** Submit code updates through pull requests to the `devel` branch.

## Coding Guidelines

Adhere to our Coding Guidelines, documented in the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/). Key areas to review include:

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`: Active development branch.
*   `stable-2.X`: Stable release branches.
*   Create a branch based on `devel` for PRs, and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup)

More information can be found on the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page.

## Roadmap

Stay informed about future releases and features:

*   [Ansible Roadmap](https://docs.ansible.com/ansible/devel/roadmap/)

## Authors & License

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and has contributions from a vast community.

Ansible is sponsored by [Red Hat, Inc.](https://www.redhat.com).

**License:**  GNU General Public License v3.0 or later ([COPYING](COPYING))