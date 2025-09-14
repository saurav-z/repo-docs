# Automate Zabbix Monitoring with Ansible: community.zabbix Collection

**Effortlessly manage your Zabbix infrastructure using the power of Ansible with the `community.zabbix` collection!**  [View the original repo on GitHub](https://github.com/ansible-collections/community.zabbix)

[![plugins](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)
[![repo-sanity](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)

[![Zabbix Agent](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_agent/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_agent/badge.svg)
[![Zabbix Server](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_server/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_server/badge.svg)
[![Zabbix Proxy](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_proxy/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_proxy/badge.svg)
[![Zabbix Web](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_web/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_web/badge.svg)
[![Zabbix Javagateway](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_javagateway/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_javagateway/badge.svg)

**Key Features:**

*   **Comprehensive Automation:** Automate Zabbix configuration and management tasks.
*   **Inventory Management:** Integrate Zabbix with your Ansible inventory using dedicated inventory scripts and plugins.
*   **Modular Approach:** Utilize a wide range of modules for tasks like host management, template configuration, and more.
*   **Ready-to-Use Roles:** Deploy and configure Zabbix agents, servers, proxies, and web interfaces quickly.
*   **Community-Driven:** Benefit from community contributions and support.

## Table of Contents

*   [About the community.zabbix Collection](#automate-zabbix-monitoring-with-ansible-communityzabbix-collection)
    *   [Introduction](#introduction)
    *   [Included Content](#included-content)
    *   [Communication](#communication)
*   [Installation](#installation)
    *   [Requirements](#requirements)
    *   [Installing from Ansible Galaxy](#installing-the-collection-from-ansible-galaxy)
*   [Usage](#usage)
*   [Supported Zabbix Versions](#supported-zabbix-versions)
*   [Collection Lifecycle and Support](#collection-life-cycle-and-support)
*   [Contributing](#contributing)
*   [License](#license)

## Introduction

The `community.zabbix` Ansible Collection provides a robust set of plugins, modules, and roles to automate the management of your Zabbix monitoring environment.  This collection simplifies complex tasks, reduces manual effort, and enables consistent Zabbix configuration across your infrastructure.

## Included Content

This collection offers a wide variety of Ansible content, including:

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

Click on the name of a plugin or module to view that content's documentation.

## Communication

*   **Ansible Forum:**
    *   [Get Help](https://forum.ansible.com/c/help/6): Get help or help others.
    *   [Posts tagged with 'zabbix'](https://forum.ansible.com/tag/zabbix): Subscribe to participate in collection-related conversations.
    *   [Social Spaces](https://forum.ansible.com/c/chat/4): Gather and interact with fellow enthusiasts.
    *   [News & Announcements](https://forum.ansible.com/c/news/5): Track project-wide announcements including social events.

*   **Ansible Bullhorn Newsletter:** Used to announce releases and important changes.
    *   [Ansible Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn)

For more information about communication, see the [Ansible communication guide](https://docs.ansible.com/ansible/devel/community/communication.html).

## Installation

### Requirements

Some of the components in this collection require additional dependencies. Review the documentation for each component you are interested in by visiting the links in the [Included content](#included-content) section.

While the various roles and modules may work with earlier versions of Python and Ansible, they are only tested and maintained against Ansible Core >= 2.16 and python >= 3.9

#### External Collections

Additional collections may be required when running the various roles.

*   `ansible.posix`: Required if using SELinux portion of any roles
*   `ansible.general`: Required if using SELinux portion of any roles
*   `ansible.netcommon`: Required when using the agent role
*   `community.mysql`: Required for the proxy or server roles if using MySQL
*   `community.postgresql`: Required for the proxy or server roles if using PostgreSQL
*   `community.windows`: Required for the agent role if installing on Windows

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

Alternatively, include it in a `requirements.yml` file and install it using:

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

Reference modules and roles using their Fully Qualified Collection Namespace (FQCN) in your playbooks:

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

Or include the collection name in the playbook's `collections` element:

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
    ansible_httpapi_validate_certs: false  # For HTTPS et to true to validate server's certificate
    ansible_zabbix_auth_key: 8ec0d52432c15c91fcafe9888500cf9a607f44091ab554dbee860f6b44fac895
  tasks:
    - name: Ensure host is monitored by Zabbix
      zabbix.zabbix_host:
        ...
```

To use Basic Authentication with the Zabbix API, add the following variables:

```
zabbix_api_http_user: "user"
zabbix_api_http_password: "password"
```

## Supported Zabbix Versions

This collection aims to support Zabbix releases with official full support from Zabbix LLC.  Check the [Zabbix Life Cycle & Release Policy](https://www.zabbix.com/life_cycle_and_release_policy) for supported versions.

Support for Zabbix LTS versions may be dropped in major collection releases, primarily affecting modules. Each role has its own support matrix; consult the documentation within the `docs/` directory for specific details.

If you encounter any version inconsistencies, feel free to open a pull request or an issue.  When submitting pull requests, ensure your changes don't break existing functionality for currently supported Zabbix releases.

## Collection Life Cycle and Support

Refer to the [RELEASE](docs/RELEASE.md) document for detailed information about the collection's lifecycle and support.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for guidance on contributing to this repository.

Join the community on [Gitter](https://gitter.im/community-zabbix/community)!

## License

This collection is licensed under the GNU General Public License v3.0 or later. See [LICENSE](LICENSE) for the full license text.