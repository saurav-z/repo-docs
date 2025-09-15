# Automate Zabbix Monitoring with Ansible: Community.Zabbix Collection

**Streamline your Zabbix infrastructure management using the Ansible `community.zabbix` collection, designed to automate tasks and enhance efficiency.**  ([View the Original Repo](https://github.com/ansible-collections/community.zabbix))

[![plugins](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg) [![repo-sanity](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)

## Key Features

*   **Comprehensive Automation:** Automate the management of Zabbix resources, including hosts, groups, templates, and more.
*   **Inventory Integration:**  Use Zabbix inventory sources and plugins for dynamic host management.
*   **Modular Design:**  Utilize a wide range of modules and roles for specific Zabbix configuration tasks.
*   **Role-Based Deployment:** Simplify complex tasks with pre-built roles for Zabbix Agent, Server, Proxy, and Web components.
*   **Community-Driven:** Benefit from the active Ansible community, with forums and social spaces for support and collaboration.

## Table of Contents

*   [Key Features](#key-features)
*   [Included Content](#included-content)
    *   [Inventory Sources](#inventory-sources)
    *   [Modules](#modules)
    *   [Roles](#roles)
*   [Installation](#installation)
    *   [Requirements](#requirements)
    *   [Installing the Collection](#installing-the-collection-from-ansible-galaxy)
*   [Usage](#usage)
*   [Supported Zabbix Versions](#supported-zabbix-versions)
*   [Collection Life Cycle and Support](#collection-life-cycle-and-support)
*   [Contributing](#contributing)
*   [License](#license)

## Included Content

This collection provides various plugins, modules, and roles to manage your Zabbix infrastructure.

### Inventory Sources

*   [zabbix](scripts/inventory/zabbix.py) - Zabbix Inventory Script
*   [zabbix_inventory](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_inventory_inventory.html) - Zabbix Ansible Inventory Plugin

### Modules

[View the full list of modules and their documentation](https://docs.ansible.com/ansible/latest/collections/community/zabbix/index.html).

### Roles

*   [Zabbix Agent](docs/ZABBIX_AGENT_ROLE.md)
*   [Zabbix Javagateway](docs/ZABBIX_JAVAGATEWAY_ROLE.md)
*   [Zabbix Proxy](docs/ZABBIX_PROXY_ROLE.md)
*   [Zabbix Server](docs/ZABBIX_SERVER_ROLE.md)
*   [Zabbix Web](docs/ZABBIX_WEB_ROLE.md)

## Installation

### Requirements

This collection requires Ansible Core >= 2.16 and Python >= 3.9.  Individual components may have additional dependencies.  See the links in the [Included Content](#included-content) section for more details.

#### External Collections

You may need to install these collections:

*   `ansible.posix`
*   `ansible.general`
*   `ansible.netcommon`
*   `community.mysql` (if using MySQL)
*   `community.postgresql` (if using PostgreSQL)
*   `community.windows` (if installing the agent on Windows)

```bash
ansible-galaxy collection install ansible.posix
ansible-galaxy collection install community.general
ansible-galaxy collection install ansible.netcommon
```

### Installing the Collection from Ansible Galaxy

Install the `community.zabbix` collection using Ansible Galaxy:

```bash
ansible-galaxy collection install community.zabbix
```

Alternatively, include it in a `requirements.yml` file:

```yaml
---
collections:
  - name: community.zabbix
    version: 4.1.1
  - name: ansible.posix
    version: 1.3.0
  - name: community.general
    version: 3.7.0
```
and install it:

```bash
ansible-galaxy collection install -r requirements.yml
```

## Usage

*For detailed usage instructions and examples, refer to the documentation linked in the [Included Content](#included-content) section.*

Use modules and roles by referencing their Fully Qualified Collection Namespace (FQCN) in your playbooks. Examples are provided in the original README.

## Supported Zabbix Versions

This collection aims to support Zabbix releases with official full support from Zabbix LLC.  Consult the [Zabbix Life Cycle & Release Policy](https://www.zabbix.com/life_cycle_and_release_policy) for details.

Support for LTS versions will be dropped with Major releases of the collection and mostly affect modules.  Each role follows its own support matrix; consult the role documentation in the `docs/` directory.

## Collection Life Cycle and Support

See the [RELEASE](docs/RELEASE.md) document for information about the collection's life cycle and support.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for details on how to contribute.

Join the [Gitter community](https://gitter.im/community-zabbix/community) for discussions.

## License

GNU General Public License v3.0 or later.  See [LICENSE](LICENSE) for the full text.