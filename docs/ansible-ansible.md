[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Simple IT Automation for Everyone

Ansible is a powerful and open-source automation platform designed to simplify IT tasks, from configuration management to application deployment.  [View the source code on GitHub](https://github.com/ansible/ansible).

## Key Features of Ansible

*   **Agentless Architecture:**  Operates over SSH, eliminating the need for agents on managed nodes, simplifying setup and reducing overhead.
*   **Configuration Management:** Automate the configuration of systems and applications with ease, ensuring consistency across your infrastructure.
*   **Application Deployment:** Streamline the deployment of applications with automated, repeatable processes, minimizing errors and downtime.
*   **Cloud Provisioning:**  Provision infrastructure on various cloud providers, enabling you to manage your cloud resources effectively.
*   **Orchestration & Task Execution:**  Orchestrate complex, multi-tier deployments and execute ad-hoc tasks efficiently across your environment.
*   **Network Automation:** Automate network device configuration and management, increasing efficiency and reducing manual errors.
*   **Human-Readable Automation:** Define infrastructure and tasks using a simple, YAML-based language, making automation accessible to both technical and non-technical users.
*   **Idempotent Operations:**  Ensures that tasks are executed only when necessary, making your automation safe and reliable.
*   **Extensible with Modules:** Ansible's modular architecture allows for extending functionality to support almost any IT task.

## Getting Started with Ansible

Get up and running quickly using `pip` or your system's package manager, following our detailed [installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html).  

For the latest features and fixes, power users and developers can explore the `devel` branch.

## Community and Support

Join the vibrant Ansible community to ask questions, get help, and collaborate with fellow users.

*   **Ansible Forum:**  Find answers, share your knowledge, and connect with others in the [Ansible forum](https://forum.ansible.com/c/help/6).
*   **Social Spaces:**  Meet and interact with fellow enthusiasts in [Social Spaces](https://forum.ansible.com/c/chat/4).
*   **News & Announcements:**  Stay informed about project-wide announcements in [News & Announcements](https://forum.ansible.com/c/news/5).
*   **Bullhorn Newsletter:**  Get release announcements and important changes through the [Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn).
*   **Community Information:** For more ways to connect and learn, see [Communicating with the Ansible community](https://docs.ansible.com/ansible/devel/community/communication.html).

## Contributing

We welcome contributions from the community!

*   **Contributor's Guide:** Read the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   **Community Information:** Explore [Community Information](https://docs.ansible.com/ansible/devel/community) for various ways to contribute, including bug reports and code submissions.
*   **Pull Requests:** Submit proposed code updates via pull requests to the `devel` branch.
*   **Collaboration:** Communicate with us before making major changes to prevent duplicate efforts.

## Development Guidelines

Find detailed information about code contributions in the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/).  Review these resources:

*   [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
*   [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)

## Branch Information

*   `devel`: Corresponds to the release actively under development.
*   `stable-2.X`: Correspond to stable releases.
*   Create a branch based on `devel` for your PR and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup).
*   See the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for details on active branches.

## Roadmap

The [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) details project plans, and provides an opportunity for community influence.

## Authors & License

*   **Created by:** [Michael DeHaan](https://github.com/mpdehaan), with contributions from thousands of users.
*   **Sponsored by:** [Red Hat, Inc.](https://www.redhat.com)
*   **License:** GNU General Public License v3.0 or later.  See [COPYING](COPYING).