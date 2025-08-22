<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework: Low-Code Web Development for Real-World Applications</h1>
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
    -
    <a href="https://github.com/frappe/frappe">View on GitHub</a>
</div>

---

## Frappe Framework: Build Powerful Web Apps Faster

Frappe Framework is a full-stack, low-code web framework built with Python and JavaScript, empowering developers to rapidly build complex and scalable web applications. Inspired by the semantic web, Frappe focuses on describing the *meaning* of data, leading to more consistent and extensible applications.

### Key Features

*   **Full-Stack Development:**  Develop both front-end and back-end using a unified framework.
*   **Built-in Admin Interface:** Save time and effort with a customizable admin dashboard for managing application data.
*   **Role-Based Permissions:**  Robust user and role management to control access and secure your applications.
*   **REST API Generation:** Automatic RESTful API generation for seamless integration with other systems.
*   **Customizable Forms and Views:** Flexible customization options using server-side scripting and client-side JavaScript.
*   **Powerful Report Builder:** Create custom reports effortlessly with a user-friendly reporting tool.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting: Frappe Cloud

For a hassle-free hosting experience, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform for deploying and managing Frappe applications. It provides installation, upgrades, monitoring, and support, allowing you to focus on development.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Self-Hosting

#### Docker

**Prerequisites:** Docker, Docker Compose, Git. Refer to [Docker Documentation](https://docs.docker.com) for setup details.

**Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Start the containers:
    ```bash
    docker compose -f pwd.yml up -d
    ```

Access your application at `http://localhost:8080`. Use the following default credentials:

*   Username: `Administrator`
*   Password: `admin`

For ARM-based Docker setups, refer to the [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) documentation.

## Development Setup

### Manual Install

Use the `bench` install script for easy dependency management (including MariaDB).  See [bench documentation](https://github.com/frappe/bench) for more details.

The script will create new passwords for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (passwords displayed and saved in `~/frappe_passwords.txt`).

### Local Development

Follow these steps to set up the repository locally:

1.  Install `bench` and start the server by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation):
    ```bash
    bench start
    ```
2.  Open a new terminal window and run:
    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```
3.  Access your application at `http://frappe.localhost:8000/app`.

## Learning and Community

*   [Frappe School](https://frappe.school) - Learn Frappe and ERPNext.
*   [Official Documentation](https://docs.frappe.io/framework) - Extensive documentation.
*   [Discussion Forum](https://discuss.frappe.io/) - Engage with the community.
*   [buildwithhussain.com](https://buildwithhussain.com) - Watch Frappe in action.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://frappe.io/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

---
<div align="center">
    <a href="https://frappe.io" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png">
            <img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies" height="28"/>
        </picture>
    </a>
</div>