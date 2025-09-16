# Automate Zabbix Monitoring with Ansible: community.zabbix Collection

**Streamline your Zabbix monitoring and infrastructure management with the powerful and versatile `community.zabbix` Ansible Collection.** ([See the original repo](https://github.com/ansible-collections/community.zabbix))

![plugins](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)
![repo-sanity](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)
![Zabbix Agent](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_agent/badge.svg)
![Zabbix Server](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_server/badge.svg)
![Zabbix Proxy](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_proxy/badge.svg)
![Zabbix Web](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_web/badge.svg)
![Zabbix Javagateway](https://github.com/ansible-collections/community.zabbix/workflows/community.zabbix.zabbix_javagateway/badge.svg)

**Key Features:**

*   **Modules:** Automate configuration of Zabbix components, including actions, hosts, templates, and more.
*   **Roles:** Easily deploy and configure Zabbix Agent, Server, Proxy, Web interface, and Java Gateway.
*   **Inventory Sources:**  Discover Zabbix infrastructure with the included inventory plugins.
*   **Comprehensive Coverage:**  Manage a wide range of Zabbix resources, from hosts and groups to media types and user configurations.
*   **Integration:** Seamlessly integrates with your existing Ansible automation workflows.

## Table of Contents

*   [Automate Zabbix Monitoring with Ansible: community.zabbix Collection](#automate-zabbix-monitoring-with-ansible-communityzabbix-collection)
    *   [Key Features](#key-features)
    *   [Included Content](#included-content)
    *   [Installation](#installation)
        *   [Requirements](#requirements)
        *   [Installing the Collection from Ansible Galaxy](#installing-the-collection-from-ansible-galaxy)
    *   [Usage](#usage)
    *   [Supported Zabbix Versions](#supported-zabbix-versions)
    *   [Collection Life Cycle and Support](#collection-life-cycle-and-support)
    *   [Contributing](#contributing)
    *   [License](#license)

## Included Content

The `community.zabbix` collection provides a rich set of modules, roles, and plugins to manage and automate your Zabbix environment.

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

Ensure you have the necessary dependencies installed before using the collection. Refer to the documentation links in the [Included content](#included-content) section for details on component-specific requirements.

This collection is tested and maintained for Ansible Core >= 2.16 and Python >= 3.9.

#### External Collections

Additional collections might be required based on the roles used.

*   `ansible.posix`: Required if using SELinux features.
*   `ansible.general`: Required if using SELinux features.
*   `ansible.netcommon`: Required when using the agent role.
*   `community.mysql`: Required for the proxy or server roles if using MySQL.
*   `community.postgresql`: Required for the proxy or server roles if using PostgreSQL.
*   `community.windows`: Required for the agent role if installing on Windows.

Install the required collections with the following commands:

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

Alternatively, define the collection in a `requirements.yml` file for easier management:

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

Install the collections from `requirements.yml`:

```bash
ansible-galaxy collection install -r requirements.yml
```

## Usage

*For detailed instructions on how to use the modules and roles, please refer to the documentation links provided in the [Included content](#included-content) section.*

Use modules and roles from this collection by referencing their Fully Qualified Collection Namespace (FQCN).  Here are a few examples:

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

If you require Basic Authentication to access the Zabbix server, add the following variables:

```
zabbix_api_http_user: "user"
zabbix_api_http_password: "password"
```

## Supported Zabbix Versions

The collection prioritizes supporting Zabbix releases with full official support from Zabbix LLC. Refer to the [Zabbix Life Cycle & Release Policy](https://www.zabbix.com/life_cycle_and_release_policy) for supported versions.

Support for Zabbix LTS versions will be dropped in major releases of the collection. Modules will mostly be affected.  Each role maintains its own support matrix; consult the documentation in the `docs/` directory.

If you encounter any version inconsistencies, please open a pull request or issue.  When submitting a pull request, ensure your changes do not break existing functionality for currently supported Zabbix releases.

## Collection Life Cycle and Support

Refer to the [RELEASE](docs/RELEASE.md) document for detailed information on the collection's life cycle and support policies.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for contribution guidelines.

Join our [Gitter community](https://gitter.im/community-zabbix/community) for discussions.

## License

Licensed under the GNU General Public License v3.0 or later. See [LICENSE](LICENSE) for the full license text.