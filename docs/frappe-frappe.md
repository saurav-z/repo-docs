<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework: Low-Code Web Development Powerhouse</h1>
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

## Build Powerful Web Applications Faster with Frappe Framework

Frappe Framework is a full-stack, low-code web application framework built with Python and JavaScript, offering a rapid development environment for real-world applications. Inspired by the semantic web, Frappe allows developers to focus on *what* applications do, not just *how* they look. Explore the framework and build complex, scalable applications with ease. Learn more at the [original Frappe Framework repository](https://github.com/frappe/frappe).

### Key Features & Benefits

*   **Full-Stack Development:** Develop both front-end and back-end using a unified framework, streamlining the development process.
*   **Rapid Application Development (RAD):** Pre-built admin interface and automatic features accelerate development.
*   **Built-in Admin Interface:** Easily manage application data with a customizable admin dashboard, saving time and effort.
*   **Role-Based Permissions:** Implement robust security with comprehensive user and role management for access control.
*   **REST API:** Automatically generate RESTful APIs for seamless integration with other systems and services.
*   **Customizable Forms & Views:** Tailor forms and views to your specific needs using server-side scripting and client-side JavaScript.
*   **Report Builder:** Empower users to create custom reports without coding, enabling data-driven decision-making.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Deployment & Hosting Options

### Managed Hosting: Frappe Cloud

Experience hassle-free Frappe application hosting with [Frappe Cloud](https://frappecloud.com). It takes care of installation, setup, upgrades, monitoring, maintenance, and support.

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

Prerequisites: docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

Run following commands:

```
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a couple of minutes, site should be accessible on your localhost port: 8080. Use below default login credentials to access the site.
- Username: Administrator
- Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The Easy Way: our install script for bench will install all dependencies (e.g. MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

To setup the repository locally follow the steps mentioned below:

1. Setup bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server
   ```
   bench start
   ```

2. In a separate terminal window, run the following commands:
   ```
   # Create a new site
   bench new-site frappe.localhost
   ```

3. Open the URL `http://frappe.localhost:8000/app` in your browser, you should see the app running

## Learn & Engage with the Frappe Community

*   [Frappe School](https://frappe.school) - Online courses to learn Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.frappe.io/framework) - Comprehensive documentation for Frappe Framework.
*   [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com) - Real-world examples and tutorials on Frappe Framework.

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