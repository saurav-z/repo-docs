<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
    <h1>Frappe Framework</h1>
    <b>Build powerful, real-world web applications quickly with Frappe, a low-code framework built on Python and JavaScript.</b>
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="License: MIT"></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Code Coverage"></a>
</div>

<div align="center">
    <img src=".github/hero-image.png" alt="Hero Image" />
</div>

<div align="center">
    <a href="https://frappe.io/framework">Website</a> |
    <a href="https://docs.frappe.io/framework">Documentation</a> |
    <a href="https://github.com/frappe/frappe">GitHub Repository</a>
</div>

## What is Frappe Framework?

Frappe Framework is a full-stack, open-source web application framework designed for rapid development of business applications. It uses Python and MariaDB on the server-side, and a tightly integrated client-side library. Frappe emphasizes a semantic approach to application development, focusing on the meaning of data to ensure consistency, extensibility, and ease of building complex applications. It is the foundation for the popular ERPNext platform.

### Key Features:

*   **Full-Stack Development:** Build both front-end and back-end components within a unified framework.
*   **Low-Code Capabilities:** Reduce development time with built-in features and automatic code generation.
*   **Integrated Admin Interface:** Manage application data and configurations using a pre-built, customizable admin dashboard.
*   **Role-Based Permissions:** Secure your application with a robust user and role management system.
*   **REST API:** Automatically generate RESTful APIs for easy integration with other systems.
*   **Customizable Forms and Views:** Tailor your user interface with flexible form and view customization options.
*   **Report Builder:** Create custom reports without the need for extensive coding.
*   **Built-in Email Integration**: Send and receive emails directly from your application.
*   **Translations Support**: Easily translate your application to multiple languages.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting: Frappe Cloud

For a hassle-free experience, consider using [Frappe Cloud](https://frappecloud.com), a user-friendly, open-source platform designed for hosting Frappe applications.  It handles installation, upgrades, monitoring, and support.

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

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run Docker Compose:

    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your site will be accessible on your localhost, port 8080.  Use the following default login credentials:

*   Username: Administrator
*   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

For detailed instructions on setting up your development environment, including bench installation and server startup, please refer to the [official installation guide](https://docs.frappe.io/framework/user/en/installation).

### Local Setup

1.  Install dependencies and start the server:
    ```bash
    bench start
    ```

2.  In a separate terminal window, run:
    ```bash
    bench new-site frappe.localhost
    ```

3.  Access your app in your browser:  `http://frappe.localhost:8000/app`

## Learning and Community

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext from various courses.
*   [Official Documentation](https://docs.frappe.io/framework) - Comprehensive documentation for Frappe Framework.
*   [Discussion Forum](https://discuss.frappe.io/) - Connect with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com) - See Frappe Framework in action.

## Contributing

We welcome contributions!  Please review the following resources:

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