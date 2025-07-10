<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework</h1>
    <b>Build powerful and scalable web applications faster with Frappe, a low-code framework built with Python and JavaScript.</b>
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg"></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj"/></a>
</div>
<div align="center">
    <img src=".github/hero-image.png" alt="Hero Image" />
</div>
<div align="center">
    <a href="https://frappe.io/framework">Website</a>
    -
    <a href="https://docs.frappe.io/framework">Documentation</a>
</div>

## About Frappe Framework

Frappe Framework is a full-stack, open-source web application framework that simplifies web app development with its low-code approach. It leverages Python and MariaDB on the server-side and a tightly integrated client-side library, making it ideal for building robust and scalable applications.  This framework, originally designed for ERPNext, offers a semantic approach to development, focusing on the meaning of data for more consistent and extensible applications.

For more information, visit the [Frappe Framework GitHub repository](https://github.com/frappe/frappe).

## Key Features

*   **Full-Stack Development:** Frappe Framework provides all the tools you need for both front-end and back-end development, enabling end-to-end application creation within a single framework.
*   **Built-in Admin Interface:** Save time and effort with a pre-built, customizable admin dashboard for easy data management and application control.
*   **Role-Based Permissions:** Implement granular access control with a comprehensive user and role management system, ensuring data security and proper user permissions.
*   **REST API Generation:** Automatically generate RESTful APIs for all your models, making integration with external systems and services straightforward.
*   **Customizable Forms and Views:** Easily tailor forms and views to your exact needs with server-side scripting and client-side JavaScript for a personalized user experience.
*   **Report Builder:** Create custom reports effortlessly using a powerful reporting tool, without writing any code.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

For a hassle-free experience, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. It handles installation, upgrades, monitoring, and support. It's a fully featured developer platform that allows you to manage and control multiple Frappe deployments.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Self Hosting

### Docker

**Prerequisites**: docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for details.

**Steps:**

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run Docker Compose:

    ```bash
    docker compose -f pwd.yml up -d
    ```

    Your site should be accessible on `localhost:8080` after a few minutes.

3.  **Default Login:**

    *   Username: Administrator
    *   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

**The Easy Way:** Use the install script for bench, which installs all dependencies (e.g., MariaDB). See [Frappe Bench Documentation](https://github.com/frappe/bench) for details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local Development

Follow these steps for local setup:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server.

    ```bash
    bench start
    ```

2.  In a separate terminal, run these commands:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open `http://frappe.localhost:8000/app` in your browser.

## Learning and Community

*   **[Frappe School](https://frappe.school):** Learn Frappe Framework and ERPNext through courses from the maintainers and community.
*   **[Official Documentation](https://docs.frappe.io/framework):** Explore the extensive documentation for Frappe Framework.
*   **[Discussion Forum](https://discuss.frappe.io/):** Engage with the Frappe Framework community.
*   **[buildwithhussain.com](https://buildwithhussain.com):** See Frappe Framework in action, building real-world web apps.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://frappe.io/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

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