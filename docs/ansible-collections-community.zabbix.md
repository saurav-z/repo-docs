# Automate Zabbix Monitoring with Ansible: community.zabbix Collection

**Effortlessly manage and configure your Zabbix infrastructure using the power of Ansible.**

[![plugins](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration)
[![repo-sanity](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity)
[![Zabbix Agent](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_agent/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_agent)
[![Zabbix Server](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_server/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_server)
[![Zabbix Proxy](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_proxy/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_proxy)
[![Zabbix Web](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_web/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_web)
[![Zabbix Javagateway](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_javagateway/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_javagateway)

**Key Features:**

*   **Modules:** Automate Zabbix configuration, including hosts, items, triggers, and more.
*   **Roles:** Simplify the deployment and management of Zabbix components such as agent, server, proxy, and web interface.
*   **Inventory Sources:** Integrate Zabbix with your Ansible inventory for dynamic host management.
*   **Seamless Integration:** Designed to integrate smoothly with Ansible workflows for efficient automation.
*   **Community-Driven:** Benefit from a community-supported collection, fostering collaboration and continuous improvement.

**[View the original repository on GitHub](https://github.com/ansible-collections/community.zabbix)**

## Included Content

This collection provides a comprehensive set of modules, roles, and inventory plugins to manage Zabbix:

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

*   Ansible Core >= 2.16 and python >= 3.9.
*   Refer to the specific module and role documentation for component-specific dependencies.

#### External Collections (if used)

*   `ansible.posix`
*   `ansible.general`
*   `ansible.netcommon`
*   `community.mysql` (for proxy or server roles with MySQL)
*   `community.postgresql` (for proxy or server roles with PostgreSQL)
*   `community.windows` (for the agent role on Windows)

Install the required collections:

```bash
ansible-galaxy collection install ansible.posix
ansible-galaxy collection install community.general
ansible-galaxy collection install ansible.netcommon
```

### Installing the Collection

Install the `community.zabbix` collection using Ansible Galaxy:

```bash
ansible-galaxy collection install community.zabbix
```

You can also include it in a `requirements.yml` file:

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

Then install with:

```bash
ansible-galaxy collection install -r requirements.yml
```

## Usage

Refer to the links in the [Included content](#included-content) section for documentation on how to use specific modules and roles.

Example usage:

```yaml
---
- name: Using Zabbix collection to install Zabbix Agent
  hosts: localhost
  roles:
    - role: community.zabbix.zabbix_agent
      zabbix_agent_server: zabbix.example.com
      ...
```

## Supported Zabbix Versions

This collection prioritizes support for Zabbix releases with official full support from Zabbix LLC. See the [Zabbix Life Cycle & Release Policy](https://www.zabbix.com/life_cycle_and_release_policy) for details.

## Collection Life Cycle and Support

See [RELEASE](docs/RELEASE.md) for information regarding the collection's life cycle and support.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for contribution guidelines.

Join our [Gitter community](https://gitter.im/community-zabbix/community) for discussions.

## License

Licensed under the GNU General Public License v3.0 or later.  See [LICENSE](LICENSE).