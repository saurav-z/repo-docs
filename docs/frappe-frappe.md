<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
    <h1>Frappe Framework</h1>
    <p><b>Build powerful, real-world web applications quickly with Frappe, a low-code framework based on Python and JavaScript.</b></p>
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="License: MIT"></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Code Coverage"/></a>
    <br>
    <a href="https://frappe.io/framework">Website</a> | <a href="https://docs.frappe.io/framework">Documentation</a> | <a href="https://github.com/frappe/frappe">View on GitHub</a>
</div>

## About Frappe Framework

Frappe Framework is a full-stack web application framework that leverages Python and MariaDB on the server-side and a tightly integrated client-side library to help you build complex applications with less code.  Inspired by the Semantic Web, Frappe allows you to define the meaning of your data, making your applications more consistent and extensible. Initially built for ERPNext, Frappe is designed for developers ready to build robust and scalable web applications.

## Key Features

*   **Full-Stack Development:**  Develop both front-end and back-end components using a single framework.
*   **Low-Code Approach:**  Reduce development time with built-in features and automatic generation of many functionalities.
*   **Built-in Admin Interface:** Customize a pre-built admin dashboard to easily manage your application's data.
*   **Role-Based Permissions:** Secure your application with a comprehensive user and role management system.
*   **REST API:** Automatically generate RESTful APIs for seamless integration with other services.
*   **Customizable Forms and Views:**  Tailor forms and views with server-side scripting and client-side JavaScript.
*   **Report Builder:** Empower users to create custom reports without writing any code.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started

### Production Setup

*   **Frappe Cloud:**  The easiest way to host Frappe applications with a fully managed, open-source platform. ([Frappe Cloud](https://frappecloud.com/))

    <div>
        <a href="https://frappecloud.com/" target="_blank">
            <picture>
                <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
                <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
            </picture>
        </a>
    </div>

### Self-Hosting

*   **Docker:**  Use Docker for a containerized setup.  Requires Docker, Docker Compose, and Git.

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    docker compose -f pwd.yml up -d
    ```

    Your site should be accessible on localhost port 8080. Use the default login:
    *   Username: Administrator
    *   Password: admin

    See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based docker setup.

## Development Setup

### Manual Install

*   **Bench:** Leverage our installation script for bench to install all dependencies (e.g., MariaDB). See [bench documentation](https://github.com/frappe/bench).

### Local Development

1.  Install Bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal:

    ```bash
    bench new-site frappe.localhost
    ```

3.  Open `http://frappe.localhost:8000/app` in your browser.

## Resources & Community

*   [Frappe School](https://frappe.school) - Learn the framework with courses by the maintainers or community.
*   [Official Documentation](https://docs.frappe.io/framework) - Comprehensive Frappe Framework documentation.
*   [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com) - See Frappe in action building real-world apps.

## Contributing

We welcome contributions!

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