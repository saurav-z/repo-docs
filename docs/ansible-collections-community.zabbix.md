# Automate Zabbix Management with Ansible: Community.Zabbix Collection

**Simplify and streamline your Zabbix monitoring infrastructure with the `community.zabbix` Ansible Collection, offering powerful automation capabilities.**  [Go to the original repo](https://github.com/ansible-collections/community.zabbix)

[![plugins](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/plugins-integration.yml)
[![repo-sanity](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/repo-sanity.yml)

## Key Features

*   **Inventory Sources:**
    *   `zabbix.py`: Zabbix Inventory Script
    *   `zabbix_inventory`: Zabbix Ansible Inventory Plugin
*   **Modules:** Automate various Zabbix tasks. (See the "Included Content" section for the comprehensive list)
*   **Roles:** Pre-built roles for common Zabbix components:

    *   [![Zabbix Agent](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_agent/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/community.zabbix.zabbix_agent.yml) Zabbix Agent
    *   [![Zabbix Server](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_server/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/community.zabbix.zabbix_server.yml) Zabbix Server
    *   [![Zabbix Proxy](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_proxy/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/community.zabbix.zabbix_proxy/badge.svg) Zabbix Proxy
    *   [![Zabbix Web](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_web/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/community.zabbix.zabbix_web.yml) Zabbix Web
    *   [![Zabbix Javagateway](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_javagateway/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/community.zabbix.zabbix_javagateway.yml) Zabbix Javagateway

## Table of Contents

-   [Automate Zabbix Management with Ansible: Community.Zabbix Collection](#automate-zabbix-management-with-ansible-communityzabbix-collection)
    *   [Key Features](#key-features)
    *   [Included Content](#included-content)
    *   [Installation](#installation)
        *   [Requirements](#requirements)
        *   [Installing the Collection from Ansible Galaxy](#installing-the-collection-from-ansible-galaxy)
    *   [Usage](#usage)
    *   [Supported Zabbix versions](#supported-zabbix-versions)
    *   [Collection life cycle and support](#collection-life-cycle-and-support)
    *   [Contributing](#contributing)
    *   [License](#license)

## Included Content

This collection provides a comprehensive suite of modules, inventory scripts, and roles for managing your Zabbix environment. Click on the name of a plugin or module to view its documentation:

  -   **Inventory Sources:**
      *   `zabbix` ([scripts/inventory/zabbix.py](scripts/inventory/zabbix.py)) - Zabbix Inventory Script
      *   `zabbix_inventory` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_inventory_inventory.html)) - Zabbix Ansible Inventory Plugin

  -   **Modules:**
      *   `zabbix_action` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_action_module.html))
      *   `zabbix_authentication` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_authentication_module.html))
      *   `zabbix_autoregister` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_autoregister_module.html))
      *   `zabbix_discovery_rule` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_discovery_rule_module.html))
      *   `zabbix_globalmacro` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_globalmacro_module.html))
      *   `zabbix_group_info` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_group_info_module.html))
      *   `zabbix_group_events_info` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_group_events_info_module.html))
      *   `zabbix_group` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_group_module.html))
      *   `zabbix_host_events_info` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_host_events_info_module.html))
      *   `zabbix_host_info` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_host_info_module.html))
      *   `zabbix_host` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_host_module.html))
      *   `zabbix_hostmacro` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_hostmacro_module.html))
      *   `zabbix_housekeeping` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_housekeeping_module.html))
      *   `zabbix_maintenance` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_maintenance_module.html))
      *   `zabbix_map` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_map_module.html))
      *   `zabbix_mediatype` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_mediatype_module.html))
      *   `zabbix_proxy_info` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_proxy_info_module.html))
      *   `zabbix_proxy` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_proxy_module.html))
      *   `zabbix_screen` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_screen_module.html))
      *   `zabbix_script` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_script_module.html))
      *   `zabbix_service` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_service_module.html))
      *   `zabbix_template_info` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_template_info_module.html))
      *   `zabbix_template` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_template_module.html))
      *   `zabbix_user_info` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_user_info_module.html))
      *   `zabbix_user` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_user_module.html))
      *   `zabbix_usergroup` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_usergroup_module.html))
      *   `zabbix_valuemap` ([Ansible Docs](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_valuemap_module.html))

  -   **Roles:**
      *   `zabbix_agent` ([docs/ZABBIX_AGENT_ROLE.md](docs/ZABBIX_AGENT_ROLE.md))
      *   `zabbix_javagateway` ([docs/ZABBIX_JAVAGATEWAY_ROLE.md](docs/ZABBIX_JAVAGATEWAY_ROLE.md))
      *   `zabbix_proxy` ([docs/ZABBIX_PROXY_ROLE.md](docs/ZABBIX_PROXY_ROLE.md))
      *   `zabbix_server` ([docs/ZABBIX_SERVER_ROLE.md](docs/ZABBIX_SERVER_ROLE.md))
      *   `zabbix_web` ([docs/ZABBIX_WEB_ROLE.md](docs/ZABBIX_WEB_ROLE.md))

## Installation

### Requirements

Ensure you have the necessary dependencies. Check the documentation of the specific modules and roles you plan to use within the "Included Content" section.

This collection is tested with Ansible Core >= 2.16 and python >= 3.9.

#### External Collections

Additional collections may be required.

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

Alternatively, include it in a `requirements.yml` file and install with:

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

*Refer to the linked documentation in the [Included Content](#included-content) section for detailed usage examples.*

Use modules and roles with their Fully Qualified Collection Namespace (FQCN):

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

Or include collection name in the playbook's `collections` element:

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

If Basic Authentication is required to access Zabbix server add following variables:
```
zabbix_api_http_user: "user"
zabbix_api_http_password: "password"
```

## Supported Zabbix versions

The collection prioritizes support for official Zabbix releases with full support from Zabbix LLC.  Consult the [Zabbix Life Cycle & Release Policy](https://www.zabbix.com/life_cycle_and_release_policy) for supported versions.

Support for Zabbix LTS versions will be dropped with Major releases of the collection and mostly affect modules. Each role is following its unique support matrix. You should always consult documentation of roles in *docs/* directory.

Please report any version inconsistencies via pull requests or issues; we aim to address them promptly.  When contributing, ensure your changes don't break existing functionality for supported Zabbix releases.

## Collection life cycle and support

Refer to the [RELEASE](docs/RELEASE.md) document for collection lifecycle and support details.

## Contributing

Review [CONTRIBUTING](CONTRIBUTING.md) for contribution guidelines.

Join our [Gitter community](https://gitter.im/community-zabbix/community) to connect with other users and contributors.

## License

Licensed under the GNU General Public License v3.0 or later.  See [LICENSE](LICENSE) for the full license text.