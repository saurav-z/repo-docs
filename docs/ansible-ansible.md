[![PyPI version](https://img.shields.io/pypi/v/ansible-core.svg)](https://pypi.org/project/ansible-core)
[![Docs badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ansible.com/ansible/latest/)
[![Chat badge](https://img.shields.io/badge/chat-IRC-brightgreen.svg)](https://docs.ansible.com/ansible/devel/community/communication.html)
[![Build Status](https://dev.azure.com/ansible/ansible/_apis/build/status/CI?branchName=devel)](https://dev.azure.com/ansible/ansible/_build/latest?definitionId=20&branchName=devel)
[![Ansible Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Ansible-silver.svg)](https://docs.ansible.com/ansible/devel/community/code_of_conduct.html)
[![Ansible mailing lists](https://img.shields.io/badge/mailing%20lists-Ansible-orange.svg)](https://docs.ansible.com/ansible/devel/community/communication.html#mailing-list-information)
[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](COPYING)
[![Ansible CII Best Practices certification](https://bestpractices.coreinfrastructure.org/projects/2372/badge)](https://bestpractices.coreinfrastructure.org/projects/2372)

# Ansible: Radically Simple IT Automation

Ansible is a powerful, open-source IT automation engine that simplifies configuration management, application deployment, cloud provisioning, and more.  Learn more and contribute to the project at the [Ansible GitHub repository](https://github.com/ansible/ansible).

## Key Features & Benefits

*   **Agentless:** Leverages SSH for communication, eliminating the need for agents and reducing complexity.
*   **Simple Setup:**  Easy to install and learn, with a minimal learning curve for rapid adoption.
*   **Parallel Execution:** Manages machines quickly and efficiently in parallel, saving time.
*   **Human-Readable:** Describes infrastructure using a simple, human-friendly language (YAML).
*   **Security Focused:** Emphasizes security and provides easy auditability and content review.
*   **Idempotent:**  Ensures that operations are performed only when necessary, avoiding unintended changes.
*   **Extensible:** Allows module development in any dynamic language, expanding its capabilities.
*   **Multi-Node Orchestration:** Simplifies complex changes such as zero-downtime rolling updates.
*   **Cloud Provisioning:** Automates infrastructure setup across various cloud providers.

## Getting Started with Ansible

### Installation

Install a released version of Ansible using `pip` or your system's package manager.

*   Refer to the [Ansible installation guide](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) for detailed instructions.

### Community & Support

*   **Community Forum:**  Ask questions, get help, and interact with the community on the [Ansible forum](https://forum.ansible.com/c/help/6).
    *   Filter and subscribe to posts using tags like: `ansible`, `ansible-core`, and `playbook`.
*   **Social Spaces:** Connect with other Ansible enthusiasts in the [Social Spaces](https://forum.ansible.com/c/chat/4).
*   **News & Announcements:** Stay up-to-date with project-wide announcements in the [News & Announcements](https://forum.ansible.com/c/news/5) section.
*   **Newsletter:** Get release announcements and important updates by subscribing to the [Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn).

### Contributing

Contribute to Ansible and help improve the project!

*   Review the [Contributor's Guide](./.github/CONTRIBUTING.md).
*   Explore [Community Information](https://docs.ansible.com/ansible/devel/community) for various contribution opportunities, including submitting bug reports and code.
*   Submit proposed code updates through a pull request to the `devel` branch.
*   Discuss significant changes beforehand to prevent duplicate efforts.

### Development

*   Review the [Developer Guide](https://docs.ansible.com/ansible/devel/dev_guide/) for coding guidelines.
    *   Specifically, see [Contributing your module to Ansible](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_checklist.html)
    *   See [Conventions, tips, and pitfalls](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_best_practices.html)
*   Create a branch based on `devel` and set up a [dev environment](https://docs.ansible.com/ansible/devel/dev_guide/developing_modules_general.html#common-environment-setup) to open a PR.

## Branch and Release Information

*   The `devel` branch represents the actively developed release.
*   `stable-2.X` branches correspond to stable releases.
*   Consult the [Ansible release and maintenance](https://docs.ansible.com/ansible/devel/reference_appendices/release_and_maintenance.html) page for details on active branches.
*   See the [Ansible Roadmap page](https://docs.ansible.com/ansible/devel/roadmap/) for planned features.

## Authors and License

*   **Created by:** [Michael DeHaan](https://github.com/mpdehaan).
*   **Sponsored by:** [Red Hat, Inc.](https://www.redhat.com)
*   **License:** GNU General Public License v3.0 or later (see [COPYING](COPYING)).