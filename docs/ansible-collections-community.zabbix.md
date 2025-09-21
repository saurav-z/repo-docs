# Automate Zabbix Monitoring with Ansible: community.zabbix Collection

**Seamlessly manage and configure your Zabbix infrastructure with the `community.zabbix` Ansible collection, streamlining your monitoring workflows.**  [View the original repository](https://github.com/ansible-collections/community.zabbix).

[![plugins](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)
[![repo-sanity](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)

[![Zabbix Agent](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_agent/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_agent/badge.svg)
[![Zabbix Server](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_server/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_server/badge.svg)
[![Zabbix Proxy](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_proxy/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_proxy/badge.svg)
[![Zabbix Web](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_web/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_web/badge.svg)
[![Zabbix Javagateway](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_javagateway/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_javagateway/badge.svg)

## Key Features

*   **Comprehensive Modules:** Manage Zabbix configurations including hosts, groups, templates, users, actions, and more.
*   **Flexible Inventory Plugins:**  Dynamically discover and manage your Zabbix infrastructure with inventory sources.
*   **Ansible Roles:** Automate the deployment and configuration of Zabbix Agent, Server, Proxy, and Web components.
*   **Simplified Configuration:**  Use Ansible's declarative approach to define your Zabbix environment.
*   **Community-Driven:** Benefit from a collection maintained and supported by the Ansible community.

## Included Content

The collection offers a variety of plugins and modules to help manage your Zabbix environment:

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

Review the included content's documentation for detailed dependency information.  This collection is tested with:

*   Ansible Core >= 2.16
*   Python >= 3.9

#### External Collections

*   `ansible.posix`
*   `ansible.general`
*   `ansible.netcommon`
*   `community.mysql` (for proxy or server with MySQL)
*   `community.postgresql` (for proxy or server with PostgreSQL)
*   `community.windows` (for agent on Windows)

### Installing from Ansible Galaxy

```bash
ansible-galaxy collection install community.zabbix
```

Alternatively, include the collection in a `requirements.yml` file:

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

and install using:

```bash
ansible-galaxy collection install -r requirements.yml
```

## Usage

*Refer to the documentation linked in the [Included content](#included-content) section for specific module and role usage.*

Use the Fully Qualified Collection Namespace (FQCN) when referencing modules and roles:

```yaml
---
- name: Using Zabbix collection to install Zabbix Agent
  hosts: localhost
  roles:
    - role: community.zabbix.zabbix_agent
      zabbix_agent_server: zabbix.example.com
      ...
```

Or, include the collection name in your playbook's `collections` element:

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
```

Basic Authentication variables:

```
zabbix_api_http_user: "user"
zabbix_api_http_password: "password"
```

## Supported Zabbix Versions

We prioritize support for Zabbix releases with official full support.  See the [Zabbix Life Cycle & Release Policy](https://www.zabbix.com/life_cycle_and_release_policy).

## Collection Life Cycle and Support

See [RELEASE](docs/RELEASE.md) for details on the collection's life cycle and support.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for information on contributing to the project.  Join our [Gitter community](https://gitter.im/community-zabbix/community) to connect with other users and contributors.

## License

GNU General Public License v3.0 or later. See [LICENSE](LICENSE) for details.