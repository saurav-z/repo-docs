# Ansible: Automate Your IT Infrastructure with Ease

Ansible is a powerful and versatile IT automation engine that simplifies configuration management, application deployment, and more.  For the latest updates and the original project, see the [Ansible GitHub repository](https://github.com/ansible/ansible).

[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

## Key Features

*   **Agentless Architecture:** Leverages SSH for secure and simple remote management, avoiding the need for agents or open ports.
*   **Configuration Management:** Automates the configuration of systems and applications across your infrastructure.
*   **Application Deployment:** Simplifies the deployment of applications with features like zero-downtime rolling updates.
*   **Cloud Provisioning:** Enables easy provisioning of resources in various cloud environments.
*   **Orchestration:** Automates complex, multi-node tasks and workflows.
*   **Human-Readable Infrastructure as Code:** Defines infrastructure in a clear, understandable language.
*   **Parallel Execution:** Manages machines quickly and efficiently in parallel.
*   **Security-Focused:** Designed with security and auditability in mind.
*   **Extensible:** Supports module development in various dynamic languages, not just Python.

## Getting Started

Install Ansible using `pip` or your preferred package manager; detailed instructions are available in the [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).

## Community and Support

*   **Community Forum:** Ask questions, get help, and engage with the Ansible community: [Ansible Forum](https://forum.ansible.com/)
    *   Find help or share your Ansible knowledge to help others.
    *   Interact with fellow enthusiasts in Social Spaces.
    *   Track project-wide news and announcements.
    *   Stay informed with the Bullhorn newsletter for release announcements.
*   **Communication:** For other ways to connect, explore [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing

Contribute to Ansible and help shape the future of automation:

*   Review the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   Explore [Community Information](https://docs.ansible.com/ansible/devel/community) for contribution options, including bug reports and code submissions.
*   Submit a pull request to the `devel` branch.
*   Discuss significant changes beforehand to coordinate efforts.

## Coding Guidelines

Adhere to the coding guidelines outlined in the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/). Specifically, see:

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   **`devel`:** Development branch with the latest features and fixes.
*   **`stable-2.X`:** Stable release branches.
*   Create a branch based on `devel` and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) if you want to open a PR.
*   Review the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for branch details.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details future plans and how to influence the project's direction.

## Authors and Sponsorship

Ansible was created by [Michael DeHaan](https://github.com/mpdehaan) and has over 5000 contributors.  Ansible is sponsored by [Red Hat, Inc.](https://www.redhat.com).

## License

Ansible is licensed under the GNU General Public License v3.0 or later.  See [COPYING](COPYING) for the full license text.