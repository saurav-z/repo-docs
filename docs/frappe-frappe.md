<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework</h1>

    **Build powerful, real-world web applications rapidly with the low-code Frappe Framework, a Python and JavaScript powerhouse.**
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
    <a href="https://github.com/frappe/frappe">Original Repository</a>
</div>

## About Frappe Framework

Frappe Framework is a full-stack, open-source web application framework that simplifies building robust, data-driven applications. It leverages Python and MariaDB on the server-side with a tightly integrated client-side library. Initially developed for ERPNext, Frappe excels at handling complex application requirements.  Inspired by the Semantic Web, Frappe focuses on defining *what* data *means*, not just *how* it's displayed, leading to more consistent and extensible applications.

## Key Features

*   ✅ **Full-Stack Development**: Develop complete applications using a single framework, from front-end to back-end.
*   ✅ **Built-in Admin Interface**:  Quickly create and customize admin dashboards for efficient data management.
*   ✅ **Role-Based Permissions**: Secure your application with a robust user and role management system.
*   ✅ **REST API**: Generate automatically a RESTful API for easy integration with other services.
*   ✅ **Customizable Forms and Views**:  Create tailored user experiences with server-side scripting and client-side JavaScript.
*   ✅ **Report Builder**: Empower users with a powerful, no-code report generation tool.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting - Frappe Cloud

For a hassle-free deployment, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. Frappe Cloud handles installation, upgrades, monitoring, and support, allowing you to focus on development.

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

1.  **Prerequisites:** Docker, docker-compose, git.
2.  **Commands:**

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    docker compose -f pwd.yml up -d
    ```

3.  **Access:** Your site should be accessible on `localhost:8080`. Use "Administrator" and "admin" for initial login.

    *   For ARM based docker setup, see [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Install

*   The easiest method is to use the install script via the `bench` utility (see [Frappe Bench](https://github.com/frappe/bench) for details), which will install all dependencies.

*   The script will generate and store new passwords for the Frappe "Administrator" user, the MariaDB root user, and the frappe user in `~/frappe_passwords.txt`.

### Local Development

1.  Set up `bench` using the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, create a new site:

    ```bash
    bench new-site frappe.localhost
    ```

3.  Access your application at `http://frappe.localhost:8000/app`.

## Learning and Community

*   📚 [Frappe School](https://frappe.school) - Courses on Frappe Framework and ERPNext.
*   📖 [Official Documentation](https://docs.frappe.io/framework) - Comprehensive framework documentation.
*   💬 [Discussion Forum](https://discuss.frappe.io/) - Connect with the Frappe community.
*   🎬 [buildwithhussain.com](https://buildwithhussain.com) - Real-world Frappe Framework examples.

## Contributing

*   📝 [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   🛡️ [Report Security Vulnerabilities](https://frappe.io/security)
*   🤝 [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   🌐 [Translations](https://crowdin.com/project/frappe)

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