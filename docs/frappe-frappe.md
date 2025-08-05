<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>
    <p><strong>Build powerful, real-world web applications quickly with Frappe Framework, a low-code platform built on Python and JavaScript.</strong></p>
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

## About Frappe Framework

Frappe Framework is a full-stack web application framework that streamlines development with its low-code approach.  It leverages Python and MariaDB on the server-side and a tightly integrated client-side library for a comprehensive development experience. Built for ERPNext, it's a powerful solution for building complex, data-driven applications.

### Key Features

*   **Full-Stack Development:**  Develop both front-end and back-end components within a single framework using Python and JavaScript.
*   **Rapid Development:**  Built-in admin interface and pre-built components reduce development time, enabling faster project completion.
*   **Role-Based Security:**  Implement granular access control with a comprehensive user and role management system.
*   **REST API Generation:**  Automated RESTful API generation for seamless integration with other systems and services.
*   **Customization Options:**  Customize forms and views extensively using server-side scripting and client-side JavaScript.
*   **Powerful Reporting:**  Create custom reports with the built-in report builder, eliminating the need for extensive coding.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started

### Production Setup

#### Frappe Cloud (Managed Hosting)

For a hassle-free experience, try [Frappe Cloud](https://frappecloud.com).  It offers a user-friendly, open-source platform for hosting Frappe applications, taking care of installation, upgrades, and maintenance.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Self Hosting

#### Docker

**Prerequisites:** Docker, Docker Compose, Git. See the [Docker Documentation](https://docs.docker.com) for setup instructions.

**Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run using Docker Compose:
    ```bash
    docker compose -f pwd.yml up -d
    ```

Your site should be accessible on localhost:8080 after a few minutes. Use the following credentials to login:

*   Username: Administrator
*   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

**The Easy Way:** Use the Frappe bench install script (see [Frappe Bench](https://github.com/frappe/bench)) which will install dependencies (e.g., MariaDB).

**Steps:**

1.  Follow the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) to set up bench and start the server.
    ```bash
    bench start
    ```
2.  In a separate terminal window:
    ```bash
    bench new-site frappe.localhost
    ```
3.  Open the URL `http://frappe.localhost:8000/app` in your browser.

## Resources for Learning and Community

1.  [Frappe School](https://frappe.school) - Access courses on Frappe Framework and ERPNext.
2.  [Official documentation](https://docs.frappe.io/framework) - Comprehensive documentation.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - See Frappe Framework in action.

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