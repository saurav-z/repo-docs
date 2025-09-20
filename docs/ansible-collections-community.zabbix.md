# Automate Zabbix Monitoring with Ansible: community.zabbix Collection

**Simplify your Zabbix infrastructure management with the `community.zabbix` Ansible Collection, providing powerful modules and roles for seamless automation.**  [View the original repository](https://github.com/ansible-collections/community.zabbix)

[![plugins](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)
[![repo-sanity](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)

## Key Features:

*   **Comprehensive Modules:** Automate Zabbix configuration tasks with modules for hosts, templates, groups, actions, and more.
*   **Pre-built Roles:**  Deploy and manage Zabbix Agent, Server, Proxy, Web interface, and Java Gateway with ease.
*   **Inventory Plugins:**  Dynamically discover and manage your Zabbix infrastructure.
*   **Simplified Management:** Reduce manual effort and ensure consistency across your Zabbix environment.
*   **Community-Driven:** Benefit from community contributions, support, and ongoing development.

## Table of Contents

*   [Key Features](#key-features)
*   [Introduction](#introduction)
*   [Included Content](#included-content)
    *   [Inventory Sources](#inventory-sources)
    *   [Modules](#modules)
    *   [Roles](#roles)
*   [Installation](#installation)
    *   [Requirements](#requirements)
    *   [Installing from Ansible Galaxy](#installing-the-collection-from-ansible-galaxy)
*   [Usage](#usage)
*   [Supported Zabbix Versions](#supported-zabbix-versions)
*   [Collection Life Cycle and Support](#collection-life-cycle-and-support)
*   [Contributing](#contributing)
*   [License](#license)
*   [Communication](#communication)

## Introduction

This repository hosts the `community.zabbix` Ansible Collection.  It provides a robust set of Ansible content for automating the configuration and management of your Zabbix monitoring infrastructure.

## Included Content

### Inventory Sources

*   **zabbix** (script):  A Zabbix Inventory Script.
*   **zabbix\_inventory** (plugin): Zabbix Ansible Inventory Plugin ([documentation](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_inventory_inventory.html)).

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

Some components in this collection have dependencies. Refer to the links in the [Included content](#included-content) section for details.

The roles and modules are tested against Ansible Core >= 2.16 and python >= 3.9.

#### External Collections

Additional collections may be required when running the various roles:

*   `ansible.posix`
*   `ansible.general`
*   `ansible.netcommon`
*   `community.mysql`
*   `community.postgresql`
*   `community.windows`

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

You can also use a `requirements.yml` file:

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
And install via: `ansible-galaxy collection install -r requirements.yml`

## Usage

*For detailed usage examples, see the documentation linked in the [Included content](#included-content) section.*

Reference modules and roles using their Fully Qualified Collection Namespace (FQCN):

```yaml
---
- name: Install Zabbix Agent using the collection
  hosts: localhost
  roles:
    - role: community.zabbix.zabbix_agent
      zabbix_agent_server: zabbix.example.com
      ...
```
Or include collection name `community.zabbix` in the playbook's `collections` element:

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

If Basic Authentication is required to access Zabbix server add the following variables:
```
zabbix_api_http_user: "user"
zabbix_api_http_password: "password"
```

## Supported Zabbix Versions

The primary focus is to support Zabbix releases with official full support from Zabbix LLC.  Check the [Zabbix Life Cycle & Release Policy](https://www.zabbix.com/life_cycle_and_release_policy) page.

Support for Zabbix LTS versions may be dropped with major collection releases. Consult the role documentation (in the *docs/* directory) for specific version support.

Please open a pull request or issue if you find inconsistencies.

## Collection Life Cycle and Support

See [RELEASE](docs/RELEASE.md) for information on the collection's life cycle and support.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for details on how to contribute.  Join our [Gitter community](https://gitter.im/community-zabbix/community).

## Communication

*   **Ansible Forum:**  Find help, participate in discussions, and stay updated ([Ansible Forum](https://forum.ansible.com/c/help/6), [Posts tagged with 'zabbix'](https://forum.ansible.com/tag/zabbix), [Social Spaces](https://forum.ansible.com/c/chat/4), [News & Announcements](https://forum.ansible.com/c/news/5)).
*   **Ansible Bullhorn Newsletter:**  Stay informed about releases and changes ([Ansible Bullhorn newsletter](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn)).
*   For more information about communication, see the [Ansible communication guide](https://docs.ansible.com/ansible/devel/community/communication.html).

## License

GNU General Public License v3.0 or later. See [LICENSE](LICENSE) for the full text.