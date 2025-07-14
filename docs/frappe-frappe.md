<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework</h1>
    <p><strong>Build powerful, data-driven web applications faster with Frappe, the low-code framework built on Python and JavaScript.</strong></p>
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg"></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj"/></a>
    <a href="https://github.com/frappe/frappe">
        <img src="https://img.shields.io/badge/GitHub-Frappe-blue?logo=github" alt="GitHub">
    </a>
</div>
<div align="center">
    <img src=".github/hero-image.png" alt="Hero Image" />
</div>
<div align="center">
    <a href="https://frappe.io/framework">Website</a>
    -
    <a href="https://docs.frappe.io/framework">Documentation</a>
</div>

## What is Frappe Framework?

Frappe Framework is a full-stack, open-source web application framework designed for building real-world applications. It uses Python and MariaDB on the server-side with a tightly integrated client-side library.  Inspired by the Semantic Web, Frappe emphasizes the meaning of data, making applications more consistent and extensible. Originally built for ERPNext, Frappe provides a robust foundation for developers to create complex, data-driven applications with less code.  See the original repository [here](https://github.com/frappe/frappe).

## Key Features

*   **Full-Stack Framework:** Develop both front-end and back-end with a single framework.
*   **Low-Code Development:** Reduces development time with built-in features and automation.
*   **Built-in Admin Interface:** Quickly manage application data with a customizable admin dashboard.
*   **Role-Based Permissions:** Implement granular user and role management for secure access control.
*   **REST API:**  Automated RESTful APIs for easy integration with other systems.
*   **Customizable Forms and Views:** Tailor user interfaces with server-side scripting and client-side JavaScript.
*   **Report Builder:** Create custom reports without writing any code.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Simplify deployments with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. Frappe Cloud handles installation, upgrades, monitoring, and support.

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

Prerequisites: docker, docker-compose, git.  Refer to [Docker Documentation](https://docs.docker.com) for more details.

Run these commands to get started:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your site at `http://localhost:8080` after a few minutes.  Use the default login credentials:

-   Username: Administrator
-   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The easiest way to install Frappe is using the install script for bench, which installs all dependencies (e.g., MariaDB). See https://github.com/frappe/bench for more details.

The script will create new passwords for the "Administrator" user, the MariaDB root user, and the frappe user (passwords are displayed and saved to `~/frappe_passwords.txt`).

### Local

To set up the repository locally, follow these steps:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, run these commands:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open the URL `http://frappe.localhost:8000/app` in your browser.

## Learning and Community

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext through courses.
*   [Official documentation](https://docs.frappe.io/framework) - Comprehensive documentation.
*   [Discussion Forum](https://discuss.frappe.io/) - Engage with the community.
*   [buildwithhussain.com](https://buildwithhussain.com) - See Frappe Framework in action.

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