# Automate Zabbix Monitoring with Ansible: community.zabbix Collection

**Seamlessly manage and automate your Zabbix infrastructure using the community.zabbix Ansible Collection.**

[![plugins](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions?query=workflow%3Aplugins-integration)
[![repo-sanity](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions?query=workflow%3Arepo-sanity)
[![Zabbix Agent](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_agent/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions?query=workflow%3Acommunity.zabbix.zabbix_agent)
[![Zabbix Server](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_server/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions?query=workflow%3Acommunity.zabbix.zabbix_server)
[![Zabbix Proxy](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_proxy/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions?query=workflow%3Acommunity.zabbix.zabbix_proxy)
[![Zabbix Web](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_web/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions?query=workflow%3Acommunity.zabbix.zabbix_web)
[![Zabbix Javagateway](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_javagateway/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions?query=workflow%3Acommunity.zabbix.zabbix_javagateway)

**Key Features:**

*   **Comprehensive Modules:** Automate Zabbix configuration and management tasks.
*   **Pre-built Roles:**  Simplify deployment and configuration of Zabbix components.
*   **Inventory Plugins:**  Dynamically discover and manage Zabbix hosts.
*   **Easy Installation:**  Install the collection via Ansible Galaxy.
*   **Community Driven:** Benefit from the support of an active community.

**[View the original repository on GitHub](https://github.com/ansible-collections/community.zabbix)**

## Table of Contents

*   [Introduction](#introduction)
*   [Included Content](#included-content)
    *   [Inventory Sources](#inventory-sources)
    *   [Modules](#modules)
    *   [Roles](#roles)
*   [Installation](#installation)
    *   [Requirements](#requirements)
    *   [Installing from Ansible Galaxy](#installing-from-ansible-galaxy)
*   [Usage](#usage)
*   [Supported Zabbix Versions](#supported-zabbix-versions)
*   [Collection Lifecycle and Support](#collection-life-cycle-and-support)
*   [Contributing](#contributing)
*   [License](#license)

## Introduction

This repository hosts the `community.zabbix` Ansible Collection, providing a suite of plugins, modules, and roles designed to streamline the automation of Zabbix monitoring and management.

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

*   [zabbix_agent](docs/ZABBIX_AGENT_ROLE.md)
*   [zabbix_javagateway](docs/ZABBIX_JAVAGATEWAY_ROLE.md)
*   [zabbix_proxy](docs/ZABBIX_PROXY_ROLE.md)
*   [zabbix_server](docs/ZABBIX_SERVER_ROLE.md)
*   [zabbix_web](docs/ZABBIX_WEB_ROLE.md)

## Installation

### Requirements

Some components in this collection require additional dependencies. Review components you are interested in by visiting links present in the [Included content](#included-content) section.

While the various roles and modules may work with earlier versions of Python and Ansible, they are only tested and maintained against Ansible Core >= 2.16 and python >= 3.9.

#### External Collections

Additional collections may be required when running the various roles.

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

### Installing from Ansible Galaxy

Install the `community.zabbix` collection using the Ansible Galaxy CLI:

```bash
ansible-galaxy collection install community.zabbix
```

Alternatively, include it in a `requirements.yml` file and install with:

```bash
ansible-galaxy collection install -r requirements.yml
```

Example `requirements.yml`:

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

## Usage

*Please note that these are not working examples. For documentation on how to use content included in this collection, refer to the links in the [Included content](#included-content) section.*

Use modules and roles with their FQCN (Fully Qualified Collection Namespace):

```yaml
---
- name: Install Zabbix Agent
  hosts: localhost
  roles:
    - role: community.zabbix.zabbix_agent
      zabbix_agent_server: zabbix.example.com
      ...
```

```yaml
---
- name: If Zabbix WebUI runs on non-default (zabbix) path, e.g. http://<FQDN>/zabbixeu
  ansible.builtin.set_fact:
    ansible_zabbix_url_path: 'zabbixeu'

- name: Manage Zabbix Server elements with username/password
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
```

```yaml
---
- name: Manage Zabbix Server elements with authentication key
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

Using the `collections` element:

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

- name: Manage Zabbix Server elements with username/password
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

- name: Manage Zabbix Server elements with authentication key
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
      zabbix.zabbix_host:
        ...
```

Add the following variables if Basic Authentication is required:
```
zabbix_api_http_user: "user"
zabbix_api_http_password: "password"
```

## Supported Zabbix Versions

The collection prioritizes supporting Zabbix releases with official full support from Zabbix LLC.  Refer to the [Zabbix Life Cycle & Release Policy](https://www.zabbix.com/life_cycle_and_release_policy) for details.

Support for Zabbix LTS versions may be dropped with Major releases of the collection.

Consult role documentation in the `docs/` directory for specific support matrices.

Report any version inconsistencies via pull requests or issues. Ensure changes do not break existing functionality for currently supported Zabbix releases.

## Collection Lifecycle and Support

See [RELEASE](docs/RELEASE.md) for collection lifecycle and support information.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for contribution guidelines.

Join the [Gitter community](https://gitter.im/community-zabbix/community).

## License

GNU General Public License v3.0 or later

See [LICENSE](LICENSE) for the full text.