# Automate Zabbix Monitoring with Ansible: community.zabbix Collection

**Simplify your Zabbix infrastructure management and accelerate your automation workflows using the `community.zabbix` Ansible collection.**  [View on GitHub](https://github.com/ansible-collections/community.zabbix)

[![plugins](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/plugins-integration.yml)
[![repo-sanity](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)](https://github.com/ansible-collections/community.zabbix/actions/workflows/repo-sanity.yml)

## Key Features

*   **Comprehensive Modules:** Manage all aspects of your Zabbix environment, from hosts and templates to users and actions.
*   **Pre-built Roles:** Quickly deploy and configure Zabbix agents, servers, proxies, and web interfaces.
*   **Inventory Sources:** Integrate Zabbix data seamlessly into your Ansible inventories.
*   **Easy Installation:** Install the collection directly from Ansible Galaxy.
*   **Community-Driven:** Benefit from a community-supported collection with active development and contributions.

## Table of Contents

-   [Key Features](#key-features)
-   [Included Content](#included-content)
-   [Installation](#installation)
    -   [Requirements](#requirements)
    -   [Installing from Ansible Galaxy](#installing-from-ansible-galaxy)
-   [Usage](#usage)
-   [Supported Zabbix Versions](#supported-zabbix-versions)
-   [Communication and Support](#communication-and-support)
-   [Contributing](#contributing)
-   [License](#license)

## Included Content

This collection provides a range of modules, roles, and inventory sources to automate your Zabbix management:

**Inventory Sources:**

*   [zabbix](scripts/inventory/zabbix.py) - Zabbix Inventory Script
*   [zabbix\_inventory](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_inventory_inventory.html) - Zabbix Ansible Inventory Plugin

**Modules:**

*   [zabbix\_action](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_action_module.html)
*   [zabbix\_authentication](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_authentication_module.html)
*   [zabbix\_autoregister](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_autoregister_module.html)
*   [zabbix\_discovery\_rule](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_discovery_rule_module.html)
*   [zabbix\_globalmacro](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_globalmacro_module.html)
*   [zabbix\_group\_info](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_group_info_module.html)
*   [zabbix\_group\_events\_info](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_group_events_info_module.html)
*   [zabbix\_group](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_group_module.html)
*   [zabbix\_host\_events\_info](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_host_events_info_module.html)
*   [zabbix\_host\_info](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_host_info_module.html)
*   [zabbix\_host](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_host_module.html)
*   [zabbix\_hostmacro](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_hostmacro_module.html)
*   [zabbix\_housekeeping](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_housekeeping_module.html)
*   [zabbix\_maintenance](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_maintenance_module.html)
*   [zabbix\_map](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_map_module.html)
*   [zabbix\_mediatype](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_mediatype_module.html)
*   [zabbix\_proxy\_info](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_proxy_info_module.html)
*   [zabbix\_proxy](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_proxy_module.html)
*   [zabbix\_screen](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_screen_module.html)
*   [zabbix\_script](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_script_module.html)
*   [zabbix\_service](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_service_module.html)
*   [zabbix\_template\_info](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_template_info_module.html)
*   [zabbix\_template](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_template_module.html)
*   [zabbix\_user\_info](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_user_info_module.html)
*   [zabbix\_user](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_user_module.html)
*   [zabbix\_usergroup](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_usergroup_module.html)
*   [zabbix\_valuemap](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_valuemap_module.html)

**Roles:**

*   [zabbix\_agent](docs/ZABBIX_AGENT_ROLE.md)
*   [zabbix\_javagateway](docs/ZABBIX_JAVAGATEWAY_ROLE.md)
*   [zabbix\_proxy](docs/ZABBIX_PROXY_ROLE.md)
*   [zabbix\_server](docs/ZABBIX_SERVER_ROLE.md)
*   [zabbix\_web](docs/ZABBIX_WEB_ROLE.md)

## Installation

### Requirements

Ensure the following before installing this collection:

*   Ansible Core >= 2.16
*   Python >= 3.9

Additionally, some roles and modules require additional dependencies.  Check the documentation for each specific component for details, linked in the [Included Content](#included-content) section.

**External Collections:**

*   `ansible.posix`:  Required if using SELinux portion of any roles
*   `ansible.general`:  Required if using SELinux portion of any roles
*   `ansible.netcommon`:  Required when using the agent role
*   `community.mysql`:  Required for the proxy or server roles if using MySQL
*   `community.postgresql`:  Required for the proxy or server roles if using PostgreSQL
*   `community.windows`:  Required for the agent role if installing on Windows

You can install external collections with:

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

Or, include it in your `requirements.yml` file:

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

And install using:
```bash
ansible-galaxy collection install -r requirements.yml
```

## Usage

*   **Note:** For detailed usage examples and documentation, refer to the links provided in the [Included Content](#included-content) section.

Use modules and roles within your Ansible playbooks using their Fully Qualified Collection Namespace (FQCN). Examples:

```yaml
---
- name: Install Zabbix Agent
  hosts: all
  roles:
    - role: community.zabbix.zabbix_agent
      zabbix_agent_server: zabbix.example.com
      ...
```

```yaml
---
- name: Manage Zabbix Server with Username/Password
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
```

Or use the collection name in the playbook's `collections` section:

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
```

If Basic Authentication is required:
```
zabbix_api_http_user: "user"
zabbix_api_http_password: "password"
```

## Supported Zabbix Versions

This collection prioritizes support for Zabbix releases with official full support from Zabbix LLC.  See [Zabbix Life Cycle & Release Policy](https://www.zabbix.com/life_cycle_and_release_policy) for details.

Support for Zabbix LTS versions will be dropped with Major releases of the collection and mostly affect modules. Each role is following its unique support matrix. You should always consult documentation of roles in *docs/* directory.

Report any version-related inconsistencies via pull requests or issues.

## Communication and Support

*   **Ansible Forum:** Get help, participate in discussions, and stay informed:
    *   [Help](https://forum.ansible.com/c/help/6)
    *   ['zabbix' tag](https://forum.ansible.com/tag/zabbix)
    *   [Social Spaces](https://forum.ansible.com/c/chat/4)
    *   [News & Announcements](https://forum.ansible.com/c/news/5)

*   **Ansible Bullhorn Newsletter:** Stay updated on releases and changes: [Ansible Bullhorn](https://docs.ansible.com/ansible/devel/community/communication.html#the-bullhorn)

*   For more information about communication, see the [Ansible communication guide](https://docs.ansible.com/ansible/devel/community/communication.html).
*   **Gitter Community:** [Gitter community](https://gitter.im/community-zabbix/community)

See [RELEASE](docs/RELEASE.md) document for more information regarding life cycle and support for the collection.

## Contributing

Contribute to the `community.zabbix` collection! See [CONTRIBUTING](CONTRIBUTING.md) for guidelines.

## License

Licensed under the GNU General Public License v3.0 or later. See [LICENSE](LICENSE) for the full text.