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
</div>

## Frappe Framework: Build Powerful Web Apps with Ease

Frappe Framework is a full-stack, low-code web application framework written in Python and JavaScript, enabling rapid development of complex and scalable applications. Inspired by the semantic web, Frappe focuses on defining application metadata, making it easy to build applications that are consistent, extensible, and designed for real-world use cases.

**[Explore the original Frappe Framework repository on GitHub](https://github.com/frappe/frappe)**

### Key Features

*   **Full-Stack Development:** Develop both front-end and back-end components within a single framework, streamlining the development process.
*   **Low-Code Approach:** Reduce development time with built-in features, pre-built components, and an intuitive admin interface.
*   **Built-in Admin Interface:** Manage application data and configurations through a customizable admin dashboard.
*   **Role-Based Permissions:** Implement granular access control with a powerful user and role management system.
*   **REST API:** Automatically generate RESTful APIs for all models, simplifying integration with other systems.
*   **Customizable Forms & Views:** Tailor forms and views using server-side scripting and client-side JavaScript to meet your specific needs.
*   **Report Builder:** Empower users to create custom reports without writing code, accelerating data analysis and insights.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Frappe Cloud offers a simple, user-friendly, and sophisticated platform to host your Frappe applications. It handles installation, upgrades, monitoring, maintenance, and support.

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

**Prerequisites:** Docker, Docker Compose, Git. Refer to [Docker Documentation](https://docs.docker.com) for details.

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Start the application using Docker Compose:

    ```bash
    docker compose -f pwd.yml up -d
    ```

    After a few minutes, your site should be accessible on `localhost:8080`.

    *   **Default Login:**
        *   Username: `Administrator`
        *   Password: `admin`

    See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

Use the Frappe install script, which installs all dependencies (e.g., MariaDB). See [Frappe Bench](https://github.com/frappe/bench) for more details.

Passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the Frappe user.

### Local Development

1.  Set up Bench following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal, create a new site:

    ```bash
    bench new-site frappe.localhost
    ```

3.  Open `http://frappe.localhost:8000/app` in your browser to view your app.

## Learning and Community

1.  [Frappe School](https://frappe.school): Learn the Frappe Framework and ERPNext through community courses.
2.  [Official Documentation](https://docs.frappe.io/framework): Comprehensive documentation for Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/): Engage with the Frappe Framework community.
4.  [buildwithhussain.com](https://buildwithhussain.com): See how Frappe Framework is used to build web apps.

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