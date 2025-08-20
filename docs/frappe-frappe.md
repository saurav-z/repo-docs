<!--
  SPDX-License-Identifier: MIT
-->
<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
    <h1>Frappe Framework</h1>
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="MIT License"></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Code Coverage"></a>
</div>
<div align="center">
    <img src=".github/hero-image.png" alt="Hero Image" />
</div>
<div align="center">
    <a href="https://frappe.io/framework">Website</a>
    -
    <a href="https://docs.frappe.io/framework">Documentation</a>
</div>

## Frappe Framework: Build Powerful Web Applications with Python and JavaScript

**Frappe Framework** is a full-stack, low-code web framework that simplifies the development of real-world applications using Python and JavaScript. Explore the original repository on [GitHub](https://github.com/frappe/frappe).

### Key Features

*   **Full-Stack Development:** Develop both the front-end and back-end of your applications within a single framework, streamlining your workflow.
*   **Low-Code Capabilities:** Frappe's design allows you to build complex applications with less code, focusing on functionality rather than boilerplate.
*   **Built-in Admin Interface:** Save time and effort with a pre-built, customizable admin dashboard for managing application data.
*   **Role-Based Permissions:** Implement robust security with a comprehensive user and role management system.
*   **REST API Generation:** Automatically generate RESTful APIs for all models, allowing easy integration with other systems.
*   **Customizable Forms and Views:** Create unique and flexible user interfaces with server-side scripting and client-side JavaScript.
*   **Report Builder:** Empower users with a powerful reporting tool to create custom reports without coding.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Simplify your deployments with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. It handles installation, upgrades, monitoring, and more.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Self Hosting

#### Docker

Prerequisites: `docker`, `docker-compose`, and `git`. Refer to the [Docker Documentation](https://docs.docker.com) for installation.

1.  Clone the repository and navigate to the `frappe_docker` directory:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run the Docker Compose file:

    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your site should be accessible on `localhost:8080`. Use the following default credentials:

*   Username: `Administrator`
*   Password: `admin`

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup instructions.

## Development Setup

### Manual Install

The easy way: use the install script for bench to install all dependencies. See [bench](https://github.com/frappe/bench) for more details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

To set up the repository locally, follow these steps:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server.

    ```bash
    bench start
    ```

2.  In a separate terminal window, run the following commands:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open the URL `http://frappe.localhost:8000/app` in your browser to see the application running.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
2.  [Official documentation](https://docs.frappe.io/framework) - Extensive documentation.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - See Frappe Framework in action.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://frappe.io/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

<br>
<br>
<div align="center">
    <a href="https://frappe.io" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png">
            <img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies" height="28"/>
        </picture>
    </a>
</div>