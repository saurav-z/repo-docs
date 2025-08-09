<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
    <h1>Frappe Framework: Build Powerful Web Applications with Ease</h1>
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
    -
    <a href="https://github.com/frappe/frappe">Original Repository</a>
</div>

## Frappe Framework: The Low-Code Powerhouse for Web Applications

Frappe Framework is a full-stack, open-source web application framework that simplifies development, allowing you to build robust and scalable applications with Python and JavaScript.  It's the engine behind ERPNext and offers a comprehensive suite of tools and features to streamline your development workflow.

### Key Features:

*   **Full-Stack Development:**  Develop both front-end and back-end applications with a single framework.
*   **Built-in Admin Interface:**  Customize and manage your application data with a pre-built admin dashboard.
*   **Role-Based Permissions:** Implement granular user and role management for secure access control.
*   **REST API (Automatic):**  Instantly integrate with other systems using automatically generated RESTful APIs.
*   **Customizable Forms and Views:** Tailor forms and views with server-side scripting and client-side JavaScript to match your needs.
*   **Powerful Report Builder:** Create custom reports without writing any code.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started

### Production Setup

Choose the best hosting solution for your project:

*   **Frappe Cloud:**  Get a simple, user-friendly, and sophisticated [open-source](https://github.com/frappe/press) platform to host Frappe applications. Frappe Cloud takes care of installation, setup, upgrades, monitoring, maintenance, and support. It is a fully featured developer platform with an ability to manage and control multiple Frappe deployments.
    <div>
        <a href="https://frappecloud.com/" target="_blank">
            <picture>
                <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
                <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
            </picture>
        </a>
    </div>

*   **Self Hosting:**

    *   **Docker:**
        *   **Prerequisites:** `docker`, `docker-compose`, `git`.  Refer to [Docker Documentation](https://docs.docker.com) for setup.
        *   **Setup:**
            ```bash
            git clone https://github.com/frappe/frappe_docker
            cd frappe_docker
            docker compose -f pwd.yml up -d
            ```
        *   Your site will be accessible on localhost:8080.
        *   **Default Login:** Administrator / admin.
        *   **ARM Setup:** See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The easy way to install Frappe: use the install script for bench, which will install all dependencies (e.g., MariaDB). See [Bench Documentation](https://github.com/frappe/bench) for more details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local Development

1.  Setup bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server
    ```bash
    bench start
    ```

2.  In a separate terminal window, run:
    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open `http://frappe.localhost:8000/app` in your browser to see the running app.

## Learning and Community

*   [Frappe School](https://frappe.school): Learn Frappe Framework and ERPNext through courses by the maintainers and community.
*   [Official Documentation](https://docs.frappe.io/framework): Extensive documentation.
*   [Discussion Forum](https://discuss.frappe.io/): Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com): See Frappe Framework in action.

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