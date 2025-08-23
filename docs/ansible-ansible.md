[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Simple IT Automation for Everyone

Ansible is a powerful, open-source automation platform that simplifies IT tasks like configuration management, application deployment, and cloud provisioning. (See the original repo on Github: [https://github.com/ansible/ansible](https://github.com/ansible/ansible))

## Key Features and Benefits

*   **Agentless Architecture:** Leverages SSH for secure and simple connections, eliminating the need for agents on managed nodes.
*   **Configuration Management:** Automates the configuration of systems and applications.
*   **Application Deployment:** Streamlines the deployment process, ensuring consistency and reducing errors.
*   **Cloud Provisioning:** Enables the automated setup and management of cloud infrastructure.
*   **Orchestration:** Manages complex multi-tier deployments and rolling updates with ease.
*   **Human-Readable Automation:** Uses YAML to describe infrastructure in a clear and concise manner.
*   **Security Focused:** Designed with security in mind, promoting easy auditability and review.
*   **Parallel Execution:** Manages machines quickly and efficiently in parallel.

## Getting Started with Ansible

### Installation

You can install Ansible using `pip` or a package manager.  Detailed installation instructions are available in the [Ansible Installation Guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

### Community and Support

*   **Ansible Forum:**  Join the [Ansible forum](https://forum.ansible.com/c/help/6) to ask questions, get help, and interact with the Ansible community.
*   **Communication Channels:**  Explore other ways to connect with the community via [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing to Ansible

The Ansible project welcomes contributions from the community!

*   **Contributor's Guide:** Review the [Contributor's Guide](./.github/CONTRIBUTING.md) for detailed information on how to contribute.
*   **Community Information:**  Learn more about contributing, reporting issues, and submitting code in the [Community Information](https://docs.ansible.com/ansible/devel/community) section.
*   **Pull Requests:**  Submit code updates through pull requests to the `devel` branch.  Discuss larger changes beforehand to ensure alignment with project goals.
*   **Coding Guidelines:**  Follow the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) and pay close attention to the [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html) and [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html) sections.

## Branch Information

*   `devel`:  The active development branch, incorporating the latest features and fixes.
*   `stable-2.X`: Branches for stable releases.
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for the current branch status.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details planned features and improvements and provides a way for you to influence the direction of the project.

## Authors & License

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and has benefited from the contributions of thousands of users. Ansible is sponsored by [Red Hat, Inc.](https://www.redhat.com) and is licensed under the GNU General Public License v3.0 or later.  See [COPYING](COPYING) for the full license text.