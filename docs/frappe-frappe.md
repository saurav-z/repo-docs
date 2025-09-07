<div align="center">
    <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
    <h1>Frappe Framework</h1>
    <p><b>Build robust, real-world web applications faster with the powerful and flexible Frappe Framework.</b></p>
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="License: MIT"></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Code Coverage"></a>
</div>

<div align="center">
    <img src=".github/hero-image.png" alt="Frappe Framework Hero Image" />
</div>

<div align="center">
    <a href="https://frappe.io/framework">Website</a> |
    <a href="https://docs.frappe.io/framework">Documentation</a> |
    <a href="https://github.com/frappe/frappe"><b>View on GitHub</b></a>
</div>

## What is Frappe Framework?

Frappe Framework is a full-stack, open-source web application framework built with Python and JavaScript. It provides a rapid development environment for building custom business applications, ERP systems, and more, featuring a tightly integrated client-side library and a database-agnostic architecture (primarily using MariaDB). The framework's design philosophy emphasizes building applications based on semantic data models, promoting consistency, and extensibility. Originally developed to power ERPNext, Frappe Framework is suitable for building complex applications with ease.

## Key Features

*   **Full-Stack Development:** Develop both front-end and back-end with a single framework.
*   **Low-Code Capabilities:** Streamline development with a built-in admin interface and customizable forms and views.
*   **Role-Based Access Control:** Manage user permissions and access with a comprehensive role-based system.
*   **Automated REST API:** Automatically generate RESTful APIs for seamless integration with other services.
*   **Customizable Reports:** Create powerful custom reports without writing code using the built-in report builder.
*   **Rapid Application Development (RAD):** Built-in tools and abstractions for accelerated application development.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting (Frappe Cloud)

For the easiest deployment, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform that handles installation, upgrades, monitoring, and maintenance.  It's a fully featured developer platform with the ability to manage and control multiple Frappe deployments.

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

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a couple of minutes, site should be accessible on your localhost port: 8080. Use below default login credentials to access the site.

*   Username: Administrator
*   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The Easy Way: our install script for bench will install all dependencies (e.g. MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

To setup the repository locally follow the steps mentioned below:

1.  Setup bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, run the following commands:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open the URL `http://frappe.localhost:8000/app` in your browser; you should see the app running.

## Learn and Contribute

*   [Frappe School](https://frappe.school): Comprehensive courses for learning Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.frappe.io/framework): Detailed documentation for the framework.
*   [Discussion Forum](https://discuss.frappe.io/): Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com): Real-world examples of Frappe Framework applications.

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