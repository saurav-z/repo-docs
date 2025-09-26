# Automate Zabbix Monitoring with Ansible: community.zabbix Collection

**Effortlessly manage your Zabbix infrastructure using Ansible, streamlining configuration and monitoring tasks.**  [View the community.zabbix Collection on GitHub](https://github.com/ansible-collections/community.zabbix)

[![plugins](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/plugins-integration.yml)
[![repo-sanity](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/repo-sanity.yml)

**Roles Overview:**
[![Zabbix Agent](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_agent/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/community.zabbix.zabbix_agent.yml)
[![Zabbix Server](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_server/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/community.zabbix.zabbix_server.yml)
[![Zabbix Proxy](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_proxy/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/community.zabbix.zabbix_proxy.yml)
[![Zabbix Web](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_web/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/community.zabbix.zabbix_web.yml)
[![Zabbix Javagateway](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_javagateway/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/community.zabbix.zabbix_javagateway.yml)

**Key Features:**

*   **Modules**: Automate Zabbix configuration tasks with modules like `zabbix_host`, `zabbix_template`, `zabbix_action`, and many more.
*   **Roles**: Streamline deployment and configuration of Zabbix components (Agent, Server, Proxy, Web Frontend, Java Gateway).
*   **Inventory Sources**: Dynamic Zabbix inventory sources for use within Ansible playbooks.
*   **Simplified Management**: Manage Zabbix elements (hosts, groups, templates, users, etc.) through easy-to-use Ansible modules.
*   **Automated Monitoring**:  Automate the monitoring of your infrastructure by dynamically creating and managing Zabbix configurations.

## Table of Contents

*   [Introduction](#introduction)
*   [Included Content](#included-content)
*   [Installation](#installation)
*   [Usage](#usage)
*   [Supported Zabbix Versions](#supported-zabbix-versions)
*   [Collection Life Cycle and Support](#collection-life-cycle-and-support)
*   [Contributing](#contributing)
*   [Communication](#communication)
*   [License](#license)

## Introduction

The `community.zabbix` Ansible Collection offers a comprehensive set of modules, roles, and plugins to automate the management of your Zabbix monitoring infrastructure. This collection streamlines the configuration, deployment, and maintenance of Zabbix elements, saving time and reducing manual effort.

## Included Content

This collection provides a wide array of modules, roles, and plugins.

*   **Inventory Sources**:

    *   [zabbix](scripts/inventory/zabbix.py) - Zabbix Inventory Script
    *   [zabbix_inventory](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_inventory_inventory.html) - Zabbix Ansible Inventory Plugin
*   **Modules**:

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
*   **Roles**:

    *   [zabbix_agent](docs/ZABBIX_AGENT_ROLE.md)
    *   [zabbix_javagateway](docs/ZABBIX_JAVAGATEWAY_ROLE.md)
    *   [zabbix_proxy](docs/ZABBIX_PROXY_ROLE.md)
    *   [zabbix_server](docs/ZABBIX_SERVER_ROLE.md)
    *   [zabbix_web](docs/ZABBIX_WEB_ROLE.md)

## Installation

### Requirements

Ensure that you meet the below requirements before proceeding with installation:

*   Ansible Core >= 2.16
*   Python >= 3.9

Further dependencies are required for certain components. Please review the linked documentation provided under the [Included Content](#included-content) section for more details.

*   **External Collections**

    *   `ansible.posix`: Required if using SELinux portion of any roles
    *   `ansible.general`: Required if using SELinux portion of any roles
    *   `ansible.netcommon`: Required when using the agent role
    *   `community.mysql`: Required for the proxy or server roles if using MySQL
    *   `community.postgresql`: Required for the proxy or server roles if using PostgreSQL
    *   `community.windows`: Required for the agent role if installing on Windows

    Install required collections using `ansible-galaxy`:

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

Alternatively, include it in a `requirements.yml` file and install all collections at once:

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

Install using: `ansible-galaxy collection install -r requirements.yml`

## Usage

*Please refer to the documentation links in the [Included content](#included-content) section for detailed usage examples.*

Use modules and roles from this collection by referencing their Fully Qualified Collection Namespace (FQCN) in your playbooks.

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

Alternatively, include `community.zabbix` in the playbook's `collections` element:

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

If Basic Authentication is required to access Zabbix server add the following variables:
```
zabbix_api_http_user: "user"
zabbix_api_http_password: "password"
```

## Supported Zabbix Versions

The primary focus is to support the latest Zabbix releases with full official support from Zabbix LLC. Refer to the [Zabbix Life Cycle & Release Policy](https://www.zabbix.com/life_cycle_and_release_policy) page for version information.

Support for Zabbix LTS versions will be dropped with Major releases of the collection. Each role maintains its individual support matrix. Always consult the documentation within each role's `docs/` directory.

Report any inconsistencies or issues by opening a pull request or issue, and we will address it promptly. Ensure that your changes do not compromise existing functionality for currently supported Zabbix releases.

## Collection Life Cycle and Support

See the [RELEASE](docs/RELEASE.md) document for details about the collection's life cycle and support.

## Contributing

Refer to [CONTRIBUTING](CONTRIBUTING.md) for information on how to contribute.

## Communication

*   **Ansible Forum:**

    *   [Get Help](https://forum.ansible.com/c/help/6): Get assistance or help others.
    *   [Posts tagged with 'zabbix'](https://forum.ansible.com/tag/zabbix): Follow collection-related conversations.
    *   [Social Spaces](https://forum.ansible.com/c/chat/4): Interact with other enthusiasts.
    *   [News & Announcements](https://forum.ansible.com/c/news/5): Track project announcements.
*   **Ansible Bullhorn Newsletter:** Used to announce releases and important changes. [More Information](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn)
*   **Gitter Community**: Feel free to join our community at [Gitter community](https://gitter.im/community-zabbix/community).

For further information on communication, refer to the [Ansible communication guide](https://docs.ansible.com/ansible/devel/community/communication.html).

## License

GNU General Public License v3.0 or later. See [LICENSE](LICENSE) for full details.