[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate Everything with Ease

**Ansible is a powerful and simple IT automation engine that streamlines configuration management, application deployment, and more.**  For more information, please visit the [Ansible website](https://ansible.com/).  Find the original repository on [GitHub](https://github.com/ansible/ansible).

## Key Features of Ansible

*   **Agentless Architecture:** No agents to install, manage, or maintain, using SSH for secure communication.
*   **Simple Setup:**  Easy to get started with a minimal learning curve.
*   **Parallel Execution:**  Manages machines quickly and in parallel, saving time and resources.
*   **Human-Readable Language:**  Uses a declarative language (YAML) to describe infrastructure, making it easy to understand and maintain.
*   **Idempotent Operations:**  Ensures that tasks are only performed if necessary, avoiding unintended changes.
*   **Extensible:** Allows module development in any dynamic language, not just Python.
*   **Comprehensive Automation:** Supports configuration management, application deployment, cloud provisioning, and more.
*   **Security-Focused:** Designed with security in mind, emphasizing auditability and reviewability.

## Getting Started with Ansible

Install a released version of Ansible using `pip` or a package manager.  For detailed installation instructions, please consult the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

Power users and developers can work with the `devel` branch, which offers the latest features and fixes. However, be aware that this branch may contain breaking changes.

## Community and Communication

Engage with the Ansible community to ask questions, find support, and share your knowledge.

*   **[Ansible Forum](https://forum.ansible.com/c/help/6):** Find help, share your expertise, and connect with other users. Explore discussions tagged with [ansible](https://forum.ansible.com/tag/ansible), [ansible-core](https://forum.ansible.com/tag/ansible-core), or [playbook](https://forum.ansible.com/tag/playbook).
*   **[Social Spaces](https://forum.ansible.com/c/chat/4):** Interact with fellow enthusiasts.
*   **[News & Announcements](https://forum.ansible.com/c/news/5):** Stay updated on project-wide announcements.
*   **[Bullhorn Newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn):** Receive release announcements and important updates.
*   **[Community Communication](https://docs.ansible.com/ansible/devel/community/communication.html)**: Learn more ways to connect.

## Contributing to Ansible

Contribute to the Ansible project and help shape the future of automation.

*   **[Contributor's Guide](./.github/CONTRIBUTING.md):** Read the guide to learn how to contribute.
*   **[Community Information](https://docs.ansible.com/ansible/devel/community):** Explore various ways to contribute, including submitting bug reports and code.
*   **Submit Pull Requests:**  Submit proposed code updates to the `devel` branch.
*   **Discuss Changes:**  Communicate with the community before making larger changes.

## Coding Guidelines

Follow these guidelines for consistent and maintainable code.  Refer to the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) for detailed information.

*   **[Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)**
*   **[Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)**

## Branch Information

*   `devel`:  The active development branch.
*   `stable-2.X`:  Stable release branches.
*   Create a branch from `devel` and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) to open a PR.
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page.

## Roadmap

See the [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) to view plans and influence the direction of Ansible.

## Authors and License

*   **Created by:** [Michael DeHaan](https://github.com/mpdehaan) with contributions from over 5000 users.
*   **Sponsored by:** [Red Hat, Inc.](https://www.redhat.com)
*   **License:** GNU General Public License v3.0 or later. See [COPYING](COPYING) for the full text.