[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Automate Your IT Infrastructure with Ease

Ansible is a powerful, open-source automation tool that simplifies IT tasks, from configuration management to application deployment.  **(Original Repo: [https://github.com/ansible/ansible](https://github.com/ansible/ansible))**

## Key Features

*   **Agentless Architecture:**  Leverages SSH for secure, agentless connections, simplifying setup and management.
*   **Configuration Management:** Automates the configuration of systems and applications across your infrastructure.
*   **Application Deployment:**  Streamlines application deployment, including zero-downtime rolling updates.
*   **Cloud Provisioning:** Automates the provisioning of cloud resources, such as virtual machines and containers.
*   **Orchestration:** Orchestrates multi-node operations and complex workflows.
*   **Ad-Hoc Task Execution:** Allows for quick and easy execution of ad-hoc tasks on remote machines.
*   **Network Automation:** Automates network device configuration and management.
*   **Human-Readable Code:** Uses a simple, YAML-based language for easy understanding and modification.
*   **Parallel Execution:** Manages machines quickly and in parallel, saving time and resources.

## Getting Started with Ansible

### Installation

You can install Ansible using `pip` or your system's package manager. Detailed installation instructions are available in the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

### Running the `devel` Branch

For access to the latest features and fixes, power users and developers can run the `devel` branch. Be aware that breaking changes may be encountered more frequently. Engagement with the Ansible community is recommended if you are using the `devel` branch.

## Community and Support

### Forums and Communication

*   **Get Help:** [Ansible Forum](https://forum.ansible.com/c/help/6) - Find answers to your questions and share your knowledge. Use tags like `ansible`, `ansible-core`, and `playbook` to filter and subscribe.
*   **Social Spaces:** [Ansible Forum](https://forum.ansible.com/c/chat/4) - Connect and interact with other Ansible enthusiasts.
*   **News & Announcements:** [Ansible Forum](https://forum.ansible.com/c/news/5) - Stay informed about project-wide updates and events.
*   **Newsletter:** [Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn) - Receive release announcements and important updates.
*   **More ways to get in touch:** [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing to Ansible

We welcome contributions!  Please review the [Contributor's Guide](./.github/CONTRIBUTING.md) and the [Community Information](https://docs.ansible.com/ansible/devel/community) pages to get started.

### Steps to Contribute:

1.  **Review the Contributor's Guide:** Understand the guidelines and best practices for contributing.
2.  **Community Discussion (for larger changes):** Discuss significant changes beforehand to prevent duplicate efforts.
3.  **Submit a Pull Request:**  Propose your code changes to the `devel` branch.

## Development Guidelines

*   **Coding Guidelines:** Refer to the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) for detailed coding guidelines.
    *   **Module Development Checklist:** [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
    *   **Best Practices:** [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)
*   **Branching:**
    *   `devel`:  Actively developed release.
    *   `stable-2.X`: Stable releases.
    *   Create a branch based on `devel` for PRs and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup).
*   **Release and Maintenance:**  [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html)

## Roadmap

*   The [Ansible Roadmap](https://docs.ansible.com/ansible/devel/roadmap/) provides details on planned features and how to influence the project's direction.

## Authors and License

*   **Created by:** [Michael DeHaan](https://github.com/mpdehaan).
*   **Contributors:**  Over 5000 contributors.
*   **Sponsored by:** [Red Hat, Inc.](https://www.redhat.com)
*   **License:**  GNU General Public License v3.0 or later ([COPYING](COPYING)).