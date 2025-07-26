<div align="center" markdown="1">
  <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
  <h1>Frappe Framework</h1>
</div>

<div align="center">
  <p><b>Build powerful, real-world web applications quickly and efficiently with the Frappe Framework, a low-code web framework built on Python and JavaScript.</b></p>
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
    <a href="https://github.com/frappe/frappe"><b>View on GitHub</b></a>
</div>

## What is Frappe Framework?

Frappe Framework is a full-stack, open-source web application framework that empowers developers to build robust web applications rapidly.  It leverages the power of Python and MariaDB on the backend and provides a tightly integrated client-side library for seamless front-end development. Designed with a semantic, metadata-driven approach, Frappe allows you to focus on the core logic of your application, making it easier to build complex and extensible systems. Originally created for ERPNext, Frappe is now a versatile tool for any project needing a powerful and efficient web application framework.

## Key Features of Frappe Framework

*   ✅ **Full-Stack Development:** Develop both front-end and back-end components using a single framework.
*   ✅ **Low-Code Capabilities:** Streamline development with built-in features and a focus on metadata-driven design.
*   ✅ **Admin Interface:** Provides a pre-built, customizable admin dashboard for easy data management.
*   ✅ **Role-Based Permissions:** Fine-grained user and role management for secure access control.
*   ✅ **REST API Generation:** Automatically generates RESTful APIs for seamless integration with other systems.
*   ✅ **Customization Options:** Flexible form and view customization with server-side and client-side scripting.
*   ✅ **Report Builder:** Create custom reports easily without writing code.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

For ease of use, consider [Frappe Cloud](https://frappecloud.com), a simple, user-friendly, and sophisticated [open-source](https://github.com/frappe/press) platform.

It handles installation, setup, upgrades, monitoring, maintenance, and support for your Frappe applications. It's a comprehensive developer platform to manage and control multiple Frappe deployments.

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

**Prerequisites:** docker, docker-compose, git.  Refer to the [Docker Documentation](https://docs.docker.com) for details on setting up Docker.

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
3.  Access your site locally on port 8080 (e.g., `localhost:8080`).
4.  Use the default login credentials:
    *   Username: `Administrator`
    *   Password: `admin`

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The easiest method is using the bench install script, which installs all dependencies (e.g., MariaDB). See [bench documentation](https://github.com/frappe/bench) for more information.

This script will generate new passwords for the Frappe "Administrator" user, the MariaDB root user, and the frappe user, displaying the passwords and saving them to `~/frappe_passwords.txt`.

### Local Setup

To set up the repository locally, follow these steps:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server
    ```bash
    bench start
    ```
2.  In a separate terminal window, execute the following commands:
    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```
3.  Open the URL `http://frappe.localhost:8000/app` in your browser, and you should see the application running.

## Learning and Community Resources

1.  [Frappe School](https://frappe.school) - Access courses created by the maintainers and community.
2.  [Official documentation](https://docs.frappe.io/framework) - Comprehensive documentation.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - See Frappe in action.

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