<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework</h1>

    **Build powerful, real-world web applications with ease using Python and JavaScript with the Frappe Framework.**
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

## Frappe Framework: Low-Code Web Framework

Frappe Framework is a full-stack web application framework designed for building complex, data-driven applications efficiently.  It leverages Python and MariaDB on the server-side with a tightly integrated client-side library. Inspired by the Semantic Web, Frappe focuses on defining the *meaning* of your data, resulting in consistent, extensible applications.  Learn more and contribute on the [official Frappe Framework GitHub repository](https://github.com/frappe/frappe).

### Key Features

*   **Full-Stack Development:** Build both front-end and back-end components within a single framework using Python and JavaScript.
*   **Built-in Admin Interface:**  Streamline development with a pre-built, customizable admin dashboard for data management.
*   **Role-Based Permissions:**  Control user access and data security with a comprehensive user and role management system.
*   **REST API Generation:**  Automatically generate a RESTful API for easy integration with other services and systems.
*   **Customizable Forms and Views:** Tailor user interfaces with flexible form and view customization, utilizing server-side scripting and client-side JavaScript.
*   **Report Builder:** Create powerful custom reports without writing code, empowering data-driven decision-making.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started

### Production Setup

*   **Frappe Cloud:**  Experience a hassle-free hosting solution with Frappe Cloud, a managed platform for Frappe applications, simplifying deployment, upgrades, and maintenance. [Visit Frappe Cloud](https://frappecloud.com/).

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

**Prerequisites:** docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the Docker Compose command:
    ```bash
    docker compose -f pwd.yml up -d
    ```

After a couple of minutes, your site should be accessible on `localhost:8080`.

Use these default credentials to log in:
*   Username: Administrator
*   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The Easy Way: Use our install script for bench, which installs all dependencies (e.g., MariaDB). See [Frappe Bench](https://github.com/frappe/bench) for more details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local Development

1.  **Setup Bench:** Follow the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:
    ```bash
    bench start
    ```
2.  **Create a New Site:** In a separate terminal, run:
    ```bash
    bench new-site frappe.localhost
    ```
3.  **Access the App:** Open `http://frappe.localhost:8000/app` in your browser.

## Resources & Community

*   **Frappe School:** Deepen your understanding with courses on Frappe Framework and ERPNext. [Visit Frappe School](https://frappe.school).
*   **Official Documentation:** Explore extensive documentation for the Frappe Framework. [Read the Documentation](https://docs.frappe.io/framework).
*   **Discussion Forum:** Engage with the Frappe community. [Join the Forum](https://discuss.frappe.io/).
*   **buildwithhussain.com:** Watch Frappe Framework being used in real-world projects. [Visit buildwithhussain.com](https://buildwithhussain.com).

## Contributing

*   **Issue Guidelines:** Understand how to contribute effectively. [View Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines).
*   **Report Security Vulnerabilities:**  Help keep Frappe secure. [Report Security Vulnerabilities](https://frappe.io/security).
*   **Pull Request Requirements:** Learn about the contribution process. [See Contribution Guidelines](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines).
*   **Translations:** Help translate Frappe into other languages. [Join Crowdin](https://crowdin.com/project/frappe).

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