# Automate Zabbix Monitoring with Ansible: community.zabbix Collection

**Effortlessly manage and automate your Zabbix infrastructure using Ansible's `community.zabbix` collection.**  [View the original repository](https://github.com/ansible-collections/community.zabbix).

[![plugins](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/plugins-integration)
[![repo-sanity](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity/badge.svg)](https://github.com/ansible-collections/community.zabbix/workflows/repo-sanity)

## Key Features

*   **Modules:** Interact with Zabbix, automating tasks such as host, template, and user management.
*   **Roles:**  Pre-built roles for quickly deploying and configuring Zabbix components like the Agent, Server, Proxy, Web interface, and Java Gateway.
*   **Inventory Sources:**  Dynamically discover and manage Zabbix hosts within your Ansible environment.

## Table of Contents

*   [Automate Zabbix Monitoring with Ansible: community.zabbix Collection](#automate-zabbix-monitoring-with-ansible-communityzabbix-collection)
    *   [Key Features](#key-features)
    *   [Included Content](#included-content)
    *   [Installation](#installation)
        *   [Requirements](#requirements)
        *   [Installing from Ansible Galaxy](#installing-from-ansible-galaxy)
    *   [Usage](#usage)
    *   [Supported Zabbix Versions](#supported-zabbix-versions)
    *   [Collection Life Cycle and Support](#collection-life-cycle-and-support)
    *   [Contributing](#contributing)
    *   [License](#license)

## Included Content

This collection offers a comprehensive suite of modules, roles, and plugins for managing Zabbix.

*   **Inventory Sources:**
    *   [zabbix](scripts/inventory/zabbix.py) - Zabbix Inventory Script
    *   [zabbix_inventory](https://docs.ansible.com/ansible/latest/collections/community/zabbix/zabbix_inventory_inventory.html) - Zabbix Ansible Inventory Plugin
*   **Modules:** (Click module name to view documentation)
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
*   **Roles:** (Click role name to view documentation)
    *   [zabbix_agent](docs/ZABBIX_AGENT_ROLE.md)
    *   [zabbix_javagateway](docs/ZABBIX_JAVAGATEWAY_ROLE.md)
    *   [zabbix_proxy](docs/ZABBIX_PROXY_ROLE.md)
    *   [zabbix_server](docs/ZABBIX_SERVER_ROLE.md)
    *   [zabbix_web](docs/ZABBIX_WEB_ROLE.md)

## Installation

### Requirements

*   Review individual component documentation (links in [Included Content](#included-content)) for specific dependency details.
*   Requires Ansible Core >= 2.16 and Python >= 3.9.

#### External Collections

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

and install with:

```bash
ansible-galaxy collection install -r requirements.yml
```

## Usage

*Refer to the documentation linked in the [Included Content](#included-content) section for specific examples and usage instructions.*

Use modules and roles by referencing their Fully Qualified Collection Namespace (FQCN):

```yaml
---
- name: Install Zabbix Agent
  hosts: localhost
  roles:
    - role: community.zabbix.zabbix_agent
      zabbix_agent_server: zabbix.example.com
      ...

- name:  Use Zabbix Collection with modules
  hosts: zabbix.example.com
  vars:
    ansible_network_os: community.zabbix.zabbix
    ansible_connection: httpapi
    ansible_httpapi_port: 80
    ansible_httpapi_use_ssl: false
    ansible_httpapi_validate_certs: false
    ansible_user: Admin
    ansible_httpapi_pass: zabbix
  tasks:
    - name: Ensure host is monitored by Zabbix
      community.zabbix.zabbix_host:
        ...
```

Or add the collection to your playbook's `collections` element:

```yaml
---
- name: Use Zabbix collection
  hosts: localhost
  collections:
    - community.zabbix

  roles:
    - role: zabbix_agent
      zabbix_agent_server: zabbix.example.com
      ...

- name: Using Zabbix collection to manage Zabbix Server's elements
  hosts: zabbix.example.com
  vars:
    ansible_network_os: community.zabbix.zabbix
    ansible_connection: httpapi
    ansible_httpapi_port: 80
    ansible_httpapi_use_ssl: false
    ansible_httpapi_validate_certs: false
    ansible_user: Admin
    ansible_httpapi_pass: zabbix
  tasks:
    - name: Ensure host is monitored by Zabbix
      zabbix.zabbix_host:
        ...
```

To use Basic Authentication:
```
zabbix_api_http_user: "user"
zabbix_api_http_password: "password"
```

## Supported Zabbix Versions

The collection prioritizes support for Zabbix releases with official full support from Zabbix LLC.  Check the [Zabbix Life Cycle & Release Policy](https://www.zabbix.com/life_cycle_and_release_policy) for version details.

Support for Zabbix LTS versions will be dropped with Major releases of the collection and mostly affect modules. Each role is following its unique support matrix. You should always consult documentation of roles in *docs/* directory.

Report any version inconsistencies or contribute to the project via pull requests or issues.

## Collection Life Cycle and Support

See the [RELEASE](docs/RELEASE.md) document for information regarding the collection's life cycle and support policies.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for contribution guidelines.  Join the [Gitter community](https://gitter.im/community-zabbix/community) for discussions.

## License

Licensed under the GNU General Public License v3.0 or later.  See [LICENSE](LICENSE) for the full license text.
```
Key improvements and SEO optimizations:

*   **Clear, concise, and keyword-rich headings:** Using keywords like "Zabbix," "Ansible," "Automation," "Collection," and "Monitoring."
*   **One-sentence hook:** Immediately grabs attention and summarizes the collection's purpose.
*   **Bulleted key features:** Highlights the core value proposition.
*   **Detailed content links:** Makes the README easier to navigate.
*   **Explicit instructions and examples:** Provides actionable steps for getting started, including collection install via galaxy.
*   **Improved formatting and readability:** Makes it easier to scan and understand.
*   **Added badges:** to the top for quick overview of current status.
*   **Included External Collections section:** Explicitly calls out the required dependencies to get the roles and modules working.
*   **Keywords used throughout:**  Ensuring that the content is optimized for search engines.