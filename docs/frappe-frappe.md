<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
    <h1>Frappe Framework: Low-Code Web Development, Empowering Real-World Applications</h1>
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
    <a href="https://github.com/frappe/frappe">GitHub Repository</a>
</div>

## About Frappe Framework

Frappe Framework is a full-stack, low-code web application framework built with Python and JavaScript, ideal for rapidly developing sophisticated applications. It uses Python and MariaDB on the server-side and has a tightly integrated client-side library. It's designed with a semantic-first approach, making your applications consistent and extensible. Frappe powers ERPNext, a comprehensive ERP system, and is perfect for building real-world applications.

**Key Features:**

*   **Full-Stack Development:** Build complete web applications with a single framework, covering both front-end and back-end development.
*   **Built-in Admin Interface:** Quickly manage application data with a pre-built, customizable admin dashboard, saving valuable development time.
*   **Role-Based Permissions:** Implement granular control over user access and permissions with a comprehensive user and role management system.
*   **REST API Generation:** Automatically generate a RESTful API for all models, simplifying integration with other systems and services.
*   **Customizable Forms and Views:** Design flexible forms and views using server-side scripting and client-side JavaScript.
*   **Report Builder:** Create powerful, custom reports without writing any code using a dedicated reporting tool.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting: Frappe Cloud

For a hassle-free hosting experience, consider [Frappe Cloud](https://frappecloud.com). This open-source platform simplifies installation, upgrades, monitoring, and maintenance. Frappe Cloud is a full-featured developer platform for managing multiple Frappe deployments.

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

**Prerequisites:** docker, docker-compose, git. Refer to [Docker Documentation](https://docs.docker.com) for details on setting up Docker.

**Steps:**

1.  Clone the repository:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
```

2.  Run the Docker Compose command:

```bash
docker compose -f pwd.yml up -d
```

Your site should be accessible on your localhost at port 8080 within a few minutes. Use the default login credentials to access the site:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

**The Easy Way:** Utilize the Frappe install script for bench to install all dependencies (e.g., MariaDB). More details are available at [https://github.com/frappe/bench](https://github.com/frappe/bench).

This script generates new passwords for the "Administrator" user, the MariaDB root user, and the frappe user. These passwords are displayed and saved to `~/frappe_passwords.txt`.

### Local

To set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

```bash
bench start
```

2.  In a separate terminal window, run these commands:

```bash
# Create a new site
bench new-site frappe.localhost
```

3.  Open the URL `http://frappe.localhost:8000/app` in your browser; the application should be running.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext through community and maintainer courses.
2.  [Official documentation](https://docs.frappe.io/framework) - Comprehensive documentation for the Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe Framework community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - Learn by example, watch Frappe Framework being used to build web apps.

## Contributing

Contribute to Frappe Framework and help improve it:

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