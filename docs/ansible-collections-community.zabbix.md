# Automate Zabbix Management with Ansible: community.zabbix Collection

**Effortlessly manage and automate your Zabbix infrastructure using the `community.zabbix` Ansible collection, streamlining your monitoring and infrastructure management tasks.**  [Explore the original repository](https://github.com/ansible-collections/community.zabbix)

[![plugins](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)
[![repo-sanity](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)
[![Zabbix Agent](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_agent/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_agent/badge.svg)
[![Zabbix Server](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_server/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_server/badge.svg)
[![Zabbix Proxy](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_proxy/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_proxy/badge.svg)
[![Zabbix Web](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_web/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_web/badge.svg)
[![Zabbix Javagateway](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_javagateway/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_javagateway/badge.svg)

**Key Features:**

*   **Comprehensive Modules:** Manage various Zabbix components including hosts, templates, groups, actions, and more.
*   **Pre-built Roles:** Simplify common tasks with ready-to-use roles for Zabbix Agent, Server, Proxy, and Web installation and configuration.
*   **Inventory Sources:** Seamlessly integrate Zabbix inventory into your Ansible workflows.
*   **Flexible Configuration:** Configure your Zabbix environment with ease using Ansible's declarative approach.
*   **Community-Driven:** Benefit from a vibrant community and actively maintained collection.

## Table of Contents

-   [Automate Zabbix Management with Ansible: community.zabbix Collection](#automate-zabbix-management-with-ansible-communityzabbix-collection)
    -   [Key Features](#key-features)
    -   [Introduction](#introduction)
    -   [Included Content](#included-content)
    -   [Installation](#installation)
    -   [Usage](#usage)
    -   [Supported Zabbix Versions](#supported-zabbix-versions)
    -   [Collection Lifecycle and Support](#collection-lifecycle-and-support)
    -   [Contributing](#contributing)
    -   [License](#license)

## Introduction

The `community.zabbix` Ansible Collection offers a suite of modules, roles, and plugins designed to automate the management of your Zabbix monitoring infrastructure.  This collection streamlines tasks such as agent deployment, server configuration, and monitoring resource management, all within the powerful Ansible framework.

## Included Content

This collection provides a rich set of Ansible content to automate your Zabbix environment:

*   **Inventory Sources:**
    *   [zabbix](scripts/inventory/zabbix.py) - Zabbix Inventory Script
    *   [zabbix_inventory](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_inventory_inventory.html) - Zabbix Ansible Inventory Plugin
*   **Modules:**
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
*   **Roles:**
    *   [zabbix_agent](docs/ZABBIX_AGENT_ROLE.md)
    *   [zabbix_javagateway](docs/ZABBIX_JAVAGATEWAY_ROLE.md)
    *   [zabbix_proxy](docs/ZABBIX_PROXY_ROLE.md)
    *   [zabbix_server](docs/ZABBIX_SERVER_ROLE.md)
    *   [zabbix_web](docs/ZABBIX_WEB_ROLE.md)

## Installation

### Requirements

*   Some components require additional dependencies; refer to the links in the [Included content](#included-content) section.
*   Tested and maintained with Ansible Core >= 2.16 and Python >= 3.9.

#### External Collections

Ensure these are installed if using associated features:

*   `ansible.posix`
*   `ansible.general`
*   `ansible.netcommon`
*   `community.mysql` (if using MySQL for proxy or server roles)
*   `community.postgresql` (if using PostgreSQL for proxy or server roles)
*   `community.windows` (if installing the agent on Windows)

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

Alternatively, use a `requirements.yml` file for installation:

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

Install with: `ansible-galaxy collection install -r requirements.yml`

## Usage

*Refer to the links in the [Included content](#included-content) for detailed documentation and working examples.*

Use modules and roles within your Ansible playbooks by referencing their Fully Qualified Collection Namespace (FQCN):

```yaml
---
- name: Install Zabbix Agent
  hosts: localhost
  roles:
    - role: community.zabbix.zabbix_agent
      zabbix_agent_server: zabbix.example.com
      ...

- name: If Zabbix WebUI runs on non-default (zabbix) path, e.g. http://<FQDN>/zabbixeu
  ansible.builtin.set_fact:
    ansible_zabbix_url_path: 'zabbixeu'

- name: Manage Zabbix Server elements (username/password)
  hosts: zabbix.example.com
  vars:
    ansible_network_os: community.zabbix.zabbix
    ansible_connection: httpapi
    ansible_httpapi_port: 80
    ansible_httpapi_use_ssl: false  # Set to true for HTTPS
    ansible_httpapi_validate_certs: false  # For HTTPS set to true to validate server's certificate
    ansible_user: Admin
    ansible_httpapi_pass: zabbix
  tasks:
    - name: Ensure host is monitored by Zabbix
      community.zabbix.zabbix_host:
        ...

- name: Manage Zabbix Server elements (authentication key)
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

Alternatively, include `community.zabbix` in your playbook's `collections` element:

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

- name: Manage Zabbix Server elements (username/password)
  hosts: zabbix.example.com
  vars:
    ansible_network_os: community.zabbix.zabbix
    ansible_connection: httpapi
    ansible_httpapi_port: 80
    ansible_httpapi_use_ssl: false  # Set to true for HTTPS
    ansible_httpapi_validate_certs: false  # For HTTPS set to true to validate server's certificate
    ansible_user: Admin
    ansible_httpapi_pass: zabbix
  tasks:
    - name: Ensure host is monitored by Zabbix
      zabbix.zabbix_host:
        ...

- name: Manage Zabbix Server elements (authentication key)
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

If Basic Authentication is required for the Zabbix API, set these variables:

```
zabbix_api_http_user: "user"
zabbix_api_http_password: "password"
```

## Supported Zabbix Versions

This collection prioritizes support for Zabbix releases with official full support from Zabbix LLC, see [Zabbix Life Cycle & Release Policy](https://www.zabbix.com/life_cycle_and_release_policy).

Support for Zabbix LTS versions may be dropped with major collection releases and will primarily affect modules. Role support matrices are detailed in the role documentation (*docs/* directory).  Report inconsistencies or submit pull requests to address version-specific issues.

## Collection Lifecycle and Support

See the [RELEASE](docs/RELEASE.md) document for details on the collection's lifecycle and support policies.

## Contributing

Contributions are welcome! See [CONTRIBUTING](CONTRIBUTING.md) for guidelines.

Join our [Gitter community](https://gitter.im/community-zabbix/community) to connect with other users and contributors.

## License

Licensed under the GNU General Public License v3.0 or later.

See [LICENSE](LICENSE) for the full text.