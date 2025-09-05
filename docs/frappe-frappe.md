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

## Frappe Framework: Build Real-World Web Applications Faster

Frappe Framework is a powerful, open-source, full-stack web application framework using Python and JavaScript, designed to accelerate your development process.  Inspired by the Semantic Web, Frappe simplifies building complex applications with a focus on data semantics and extensibility.  Explore the original repository on [GitHub](https://github.com/frappe/frappe).

### Key Features of Frappe Framework:

*   **Full-Stack Development:** Build both front-end and back-end applications using a single framework.
*   **Low-Code Approach:** Reduce development time with built-in features and a focus on configuration over coding.
*   **Built-in Admin Interface:** Manage application data with a pre-built, customizable admin dashboard.
*   **Role-Based Permissions:** Implement granular user and role management for secure access control.
*   **REST API Generation:** Automatically generate RESTful APIs for seamless integration with other systems.
*   **Customizable Forms and Views:** Tailor forms and views with server-side scripting and client-side JavaScript.
*   **Reporting Tools:** Empower users with a powerful report builder to create custom reports without code.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting: Frappe Cloud

Simplify deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting and managing Frappe applications. It handles installation, upgrades, monitoring, and support.

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

### Local Setup:

1.  Setup bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server
    ```
    bench start
    ```
2.  In a separate terminal window, run the following commands:
    ```
    # Create a new site
    bench new-site frappe.localhost
    ```
3.  Open the URL `http://frappe.localhost:8000/app` in your browser, you should see the app running

## Learning and Community

*   [Frappe School](https://frappe.school): Learn Frappe Framework and ERPNext through courses.
*   [Official Documentation](https://docs.frappe.io/framework): Comprehensive framework documentation.
*   [Discussion Forum](https://discuss.frappe.io/): Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com): See Frappe Framework in action building web apps.

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