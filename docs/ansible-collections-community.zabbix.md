# Automate Zabbix Monitoring with Ansible: community.zabbix Collection

**Simplify your Zabbix infrastructure management with the `community.zabbix` Ansible Collection, providing robust modules and roles for efficient automation.**  [Explore the original repository](https://github.com/ansible-collections/community.zabbix).

![plugins](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg) ![repo-sanity](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)

Roles:

![Zabbix Agent](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_agent/badge.svg) ![Zabbix Server](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_server/badge.svg) ![Zabbix Proxy](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_proxy/badge.svg) ![Zabbix Web](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_web/badge.svg) ![Zabbix Javagateway](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_javagateway/badge.svg)

**Key Features:**

*   **Comprehensive Modules:** Manage Zabbix configurations for actions, authentication, discovery rules, global macros, groups, hosts, items, templates, users, and more.
*   **Pre-built Roles:** Automate Zabbix agent, server, proxy, web interface, and Java gateway deployments and configurations.
*   **Inventory Integration:**  Use Zabbix Inventory Sources and plugins to dynamically discover and manage your infrastructure.
*   **Ansible Automation:** Leverage the power of Ansible to automate repetitive tasks, reducing manual effort and improving efficiency.
*   **Community-Driven:** Benefit from a community-supported collection with ongoing updates and improvements.

## Table of Contents

*   [Automate Zabbix Monitoring with Ansible: community.zabbix Collection](#automate-zabbix-monitoring-with-ansible-communityzabbix-collection)
    *   [Key Features](#key-features)
    *   [Introduction](#introduction)
    *   [Included Content](#included-content)
        *   [Inventory Sources](#inventory-sources)
        *   [Modules](#modules)
        *   [Roles](#roles)
    *   [Installation](#installation)
        *   [Requirements](#requirements)
        *   [Installing the Collection from Ansible Galaxy](#installing-the-collection-from-ansible-galaxy)
    *   [Usage](#usage)
    *   [Supported Zabbix Versions](#supported-zabbix-versions)
    *   [Collection Life Cycle and Support](#collection-life-cycle-and-support)
    *   [Contributing](#contributing)
    *   [License](#license)

## Introduction

The `community.zabbix` Ansible Collection provides a suite of modules, roles, and plugins designed to automate and streamline the management of Zabbix monitoring environments.  It allows you to automate configuration, deployment, and management tasks, saving time and reducing the potential for manual errors.

## Included Content

### Inventory Sources

*   [zabbix](scripts/inventory/zabbix.py) - Zabbix Inventory Script
*   [zabbix_inventory](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_inventory_inventory.html) - Zabbix Ansible Inventory Plugin

### Modules

*   [zabbix_action](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_action_module.html)
*   [zabbix_authentication](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_authentication_module.html)
*   [zabbix_autoregister](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_autoregister_module.html)
*   [zabbix_discovery_rule](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_discovery_rule_module.html)
*   [zabbix_globalmacro](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_globalmacro_module.html)
*   [zabbix_group_info](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_group_info_module.html)
*   [zabbix_group_events_info](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_group_events_info_module.html)
*   [zabbix_group](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_group_module.html)
*   [zabbix_host_events_info](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_host_events_info_module.html)
*   [zabbix_host_info](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_host_info_module.html)
*   [zabbix_host](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_host_module.html)
*   [zabbix_hostmacro](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_hostmacro_module.html)
*   [zabbix_housekeeping](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_housekeeping_module.html)
*   [zabbix_maintenance](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_maintenance_module.html)
*   [zabbix_map](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_map_module.html)
*   [zabbix_mediatype](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_mediatype_module.html)
*   [zabbix_proxy_info](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_proxy_info_module.html)
*   [zabbix_proxy](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_proxy_module.html)
*   [zabbix_screen](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_screen_module.html)
*   [zabbix_script](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_script_module.html)
*   [zabbix_service](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_service_module.html)
*   [zabbix_template_info](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_template_info_module.html)
*   [zabbix_template](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_template_module.html)
*   [zabbix_user_info](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_user_info_module.html)
*   [zabbix_user](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_user_module.html)
*   [zabbix_usergroup](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_usergroup_module.html)
*   [zabbix_valuemap](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_valuemap_module.html)

### Roles

*   [zabbix\_agent](docs/ZABBIX\_AGENT\_ROLE.md)
*   [zabbix\_javagateway](docs/ZABBIX\_JAVAGATEWAY\_ROLE.md)
*   [zabbix\_proxy](docs/ZABBIX\_PROXY\_ROLE.md)
*   [zabbix\_server](docs/ZABBIX\_SERVER\_ROLE.md)
*   [zabbix\_web](docs/ZABBIX\_WEB\_ROLE.md)

## Installation

### Requirements

Before installing the collection, review the requirements for the specific components you intend to use, linked in the [Included content](#included-content) section.  Ensure you have the necessary versions of Python and Ansible.

*   Ansible Core >= 2.16
*   python >= 3.9

#### External Collections

*   `ansible.posix`:  Required if using SELinux portion of any roles
*   `ansible.general`:  Required if using SELinux portion of any roles
*   `ansible.netcommon`:  Required when using the agent role
*   `community.mysql`:  Required for the proxy or server roles if using MySQL
*   `community.postgresql`:  Required for the proxy or server roles if using PostgreSQL
*   `community.windows`:  Required for the agent role if installing on Windows

```bash
ansible-galaxy collection install ansible.posix
ansible-galaxy collection install community.general
ansible-galaxy collection install ansible.netcommon
```

### Installing the Collection from Ansible Galaxy

Install the `community.zabbix` collection using the Ansible Galaxy CLI:

```bash
ansible-galaxy collection install community.zabbix
```

Alternatively, include the collection in a `requirements.yml` file for bulk installation:

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

Install the collections:

```bash
ansible-galaxy collection install -r requirements.yml
```

## Usage

*For detailed usage examples and documentation, refer to the links in the [Included content](#included-content) section.*

Use modules and roles from this collection by referencing their Fully Qualified Collection Namespace (FQCN):

```yaml
---
- name: Using Zabbix collection to install Zabbix Agent
  hosts: localhost
  roles:
    - role: community.zabbix.zabbix_agent
      zabbix_agent_server: zabbix.example.com
      ...

- name: If Zabbix WebUI runs on non-default (zabbix) path, e.g. http://<FQDN>/zabbixeu
  ansible.builtin.set_fact:
    ansible_zabbix_url_path: 'zabbixeu'

- name: Using Zabbix collection to manage Zabbix Server's elements with username/password
  hosts: zabbix.example.com
  vars:
    ansible_network_os: community.zabbix.zabbix
    ansible_connection: httpapi
    ansible_httpapi_port: 80
    ansible_httpapi_use_ssl: false  # Set to true for HTTPS
    ansible_httpapi_validate_certs: false  # For HTTPS et to true to validate server's certificate
    ansible_user: Admin
    ansible_httpapi_pass: zabbix
  tasks:
    - name: Ensure host is monitored by Zabbix
      community.zabbix.zabbix_host:
        ...

- name: Using Zabbix collection to manage Zabbix Server's elements with authentication key
  hosts: zabbix.example.net
  vars:
    ansible_network_os: community.zabbix.zabbix
    ansible_connection: httpapi
    ansible_httpapi_port: 80
    ansible_httpapi_use_ssl: false  # Set to true for HTTPS
    ansible_httpapi_validate_certs: false  # For HTTPS set to true to validate server's certificate
    ansible_zabbix_auth_key: 8ec0d52432c15c91fcafe9888500cf9a607f44091ab554dbee860f6b44fac895
  tasks:
    - name: Ensure host is monitored by Zabbix
      community.zabbix.zabbix_host:
        ...
```

Alternatively, include the collection name in your playbook's `collections` element:

```yaml
---
- name: Using Zabbix collection
  hosts: localhost
  collections:
    - community.zabbix

  roles:
    - role: zabbix_agent
      zabbix_agent_server: zabbix.example.com
      ...

- name: Using Zabbix collection to manage Zabbix Server's elements with username/password
  hosts: zabbix.example.com
  vars:
    ansible_network_os: community.zabbix.zabbix
    ansible_connection: httpapi
    ansible_httpapi_port: 80
    ansible_httpapi_use_ssl: false  # Set to true for HTTPS
    ansible_httpapi_validate_certs: false  # For HTTPS et to true to validate server's certificate
    ansible_user: Admin
    ansible_httpapi_pass: zabbix
  tasks:
    - name: Ensure host is monitored by Zabbix
      zabbix.zabbix_host:
        ...

- name: Using Zabbix collection to manage Zabbix Server's elements with authentication key
  hosts: zabbix.example.net
  vars:
    ansible_network_os: community.zabbix.zabbix
    ansible_connection: httpapi
    ansible_httpapi_port: 80
    ansible_httpapi_use_ssl: false  # Set to true for HTTPS
    ansible_httpapi_validate_certs: false  # For HTTPS set to true to validate server's certificate
    ansible_zabbix_auth_key: 8ec0d52432c15c91fcafe9888500cf9a607f44091ab554dbee860f6b44fac895
  tasks:
    - name: Ensure host is monitored by Zabbix
      community.zabbix.zabbix_host:
        ...
```

If Basic Authentication is required to access Zabbix server, add these variables:
```
zabbix_api_http_user: "user"
zabbix_api_http_password: "password"
```

## Supported Zabbix Versions

The primary focus is to support Zabbix releases that have official full support from Zabbix LLC. Consult the [Zabbix Life Cycle & Release Policy](https://www.zabbix.com/life_cycle_and_release_policy) for version details.

Support for Zabbix LTS versions may be dropped with Major releases of the collection.  Always refer to the documentation of the roles in the *docs/* directory for specific support matrices.

If you encounter any version inconsistencies, please submit a pull request or issue, and the community will address it. When submitting pull requests, ensure existing functionality is maintained for the currently supported Zabbix releases.

## Collection Life Cycle and Support

See [RELEASE](docs/RELEASE.md) for information on the collection's life cycle and support.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for contributing guidelines.

Join the [Gitter community](https://gitter.im/community-zabbix/community) for discussions.

## License

GNU General Public License v3.0 or later

See [LICENSE](LICENSE) for the full license text.