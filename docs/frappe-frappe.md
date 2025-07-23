<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework</h1>
    **Build powerful, real-world web applications quickly and efficiently with the low-code Frappe Framework, powered by Python and JavaScript.**
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

## What is Frappe Framework?

Frappe Framework is a full-stack, low-code web application framework built using Python and JavaScript, alongside MariaDB.  It offers a streamlined approach to building complex web applications with a focus on metadata and semantics.  It's the backbone of the popular ERPNext system and designed for developers seeking a powerful, extensible framework.

**[Explore the Frappe Framework on GitHub](https://github.com/frappe/frappe)**

### Key Features

*   **Full-Stack Development:** Develop both the front-end and back-end of your applications using a unified framework, simplifying development and maintenance.
*   **Low-Code Capabilities:** Reduce development time with built-in features, including a customizable admin interface and automatically generated REST APIs.
*   **Admin Interface:** Benefit from a pre-built and customizable admin dashboard, enabling easy data management.
*   **Role-Based Permissions:** Implement robust security with a comprehensive user and role management system to control access and permissions.
*   **REST API Generation:** Automatically generate RESTful APIs for all your models, facilitating seamless integration with other systems.
*   **Customization:** Leverage flexible form and view customization using server-side scripting and client-side JavaScript.
*   **Report Builder:** Empower users with a powerful reporting tool to create custom reports without coding.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started

### Production Setup

Choose a hosting option for your Frappe applications:

*   **Frappe Cloud:**  A user-friendly, managed hosting platform for Frappe applications. Frappe Cloud takes care of installation, upgrades, monitoring, and support, so you can focus on your app.  [Visit Frappe Cloud](https://frappecloud.com/)

### Development Setup

Here's how to set up a local development environment:

#### Docker

1.  **Prerequisites:**  Ensure you have Docker, Docker Compose, and Git installed. See the [Docker Documentation](https://docs.docker.com/) for details.
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
3.  **Run Docker Compose:**
    ```bash
    docker compose -f pwd.yml up -d
    ```
4.  Access your site on `http://localhost:8080`.  Use the default credentials:
    *   Username: `Administrator`
    *   Password: `admin`

*   **ARM Architecture:** See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

#### Manual Install

Install dependencies using the bench install script:

```bash
# Install bench (see https://github.com/frappe/bench for details)
# This installs MariaDB and other dependencies
bench init frappe-bench
cd frappe-bench
bench new-site frappe.localhost
bench start
```
The script will generate passwords for the Frappe "Administrator" user, MariaDB root, and the frappe user, which are saved to `~/frappe_passwords.txt`.

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

*   **Frappe School:**  Learn the framework and ERPNext through courses by the maintainers and community.  [Visit Frappe School](https://frappe.school)
*   **Official Documentation:** Comprehensive documentation to guide you. [Visit the Documentation](https://docs.frappe.io/framework)
*   **Discussion Forum:** Engage with other Frappe developers and get support. [Visit the Forum](https://discuss.frappe.io/)
*   **Build with Hussain:** Watch Frappe in action to build real-world apps.  [Visit Build with Hussain](https://buildwithhussain.com)

## Contributing

*   **Issue Guidelines:**  Understand the issue submission process. [See Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   **Report Security Vulnerabilities:**  Report any security concerns. [Report Security Vulnerabilities](https://frappe.io/security)
*   **Contribution Guidelines:** Learn about making pull requests. [See Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   **Translations:**  Help translate Frappe into other languages. [Contribute Translations](https://crowdin.com/project/frappe)

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