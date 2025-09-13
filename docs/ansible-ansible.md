[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Radically Simple IT Automation

**Ansible** is a powerful, open-source automation platform that simplifies IT tasks like configuration management, application deployment, and cloud provisioning.  [Visit the original repository on GitHub](https://github.com/ansible/ansible).

## Key Features

*   **Agentless Architecture:** Leverages SSH for secure and agentless management, minimizing overhead.
*   **Configuration Management:** Automates configuration across your infrastructure.
*   **Application Deployment:** Streamlines the deployment of applications with ease.
*   **Cloud Provisioning:** Efficiently provisions resources across various cloud platforms.
*   **Orchestration:** Coordinates complex multi-node operations.
*   **Ad-hoc Task Execution:** Executes commands on remote systems quickly.
*   **Network Automation:** Automates network device configuration and management.
*   **Human-Readable Automation:** Uses a simple, YAML-based language for easy understanding and modification.
*   **Parallel Execution:** Manages machines quickly and efficiently.
*   **Security Focused:** Designed with security and auditability in mind.

## Core Design Principles

Ansible is built on the following principles:

*   **Simple Setup:**  Minimal learning curve and straightforward installation.
*   **Speed and Efficiency:**  Manages machines rapidly and in parallel.
*   **Agentless Approach:**  Uses SSH, avoiding the need for custom agents.
*   **Human-Friendly Language:**  Uses a declarative language that is easy to read and write.
*   **Security First:** Prioritizes security and easy auditing.
*   **Instant Remote Machine Management:**  No bootstrapping required for new remote machines.
*   **Flexible Module Development:**  Allows module development in any dynamic language.
*   **Non-Root Usability:** Usable as a non-root user.
*   **Ease of Use:** Designed to be the most user-friendly IT automation system available.

## Getting Started with Ansible

Install Ansible using `pip` or your preferred package manager.  Detailed installation instructions are available in the [Ansible Installation Guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

## Communication and Community

Connect with the Ansible community for support, discussions, and contributions:

*   **Forum:** [Ansible Forum](https://forum.ansible.com/c/help/6) for asking questions and sharing knowledge.
*   **Social Spaces:** [Ansible Social Spaces](https://forum.ansible.com/c/chat/4) for community interaction.
*   **News & Announcements:** [Ansible News & Announcements](https://forum.ansible.com/c/news/5) for project updates.
*   **Bullhorn Newsletter:** [The Bullhorn Newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn) for release announcements.
*   **More Ways to Connect:** [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing to Ansible

We welcome contributions!  Here's how you can get involved:

*   **Contributor's Guide:** Explore the [Contributor's Guide](./.github/CONTRIBUTING.md) for guidance.
*   **Community Information:** Read [Community Information](https://docs.ansible.com/ansible/devel/community) for detailed contribution options.
*   **Submit a Pull Request:** Propose code updates to the `devel` branch.
*   **Discuss Large Changes:** Contact us before significant modifications to coordinate efforts.

## Coding Guidelines

Review our Coding Guidelines in the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/):

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`: The active development branch.
*   `stable-2.X`: Branches for stable releases.
*   Create a branch based on `devel` to submit PRs.  Set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup).
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) for more on active branches.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details planned features and how to influence the roadmap.

## Authors

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and has contributions from thousands of users.

Sponsored by [Red Hat, Inc.](https://www.redhat.com)

## License

GNU General Public License v3.0 or later

See [COPYING](COPYING) for the full license text.