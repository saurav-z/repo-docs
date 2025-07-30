<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework</h1>
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

## Frappe Framework: Build Powerful Web Applications with Ease

**Frappe Framework** is a low-code, full-stack web application framework built with Python and JavaScript, empowering developers to create real-world applications efficiently. Inspired by the semantic web, Frappe simplifies the development of complex applications by focusing on the meaning and structure of data.

[View the original repository on GitHub](https://github.com/frappe/frappe)

### Key Features

*   **Full-Stack Framework:** Develop both front-end and back-end components within a single, integrated framework.
*   **Built-in Admin Interface:** Get a head start with a customizable admin dashboard for managing data and application settings.
*   **Role-Based Permissions:** Implement fine-grained access control with a robust user and role management system.
*   **REST API:** Automatically generate RESTful APIs for seamless integration with other services and systems.
*   **Customizable Forms and Views:** Tailor user interfaces with server-side scripting and client-side JavaScript for a perfect fit.
*   **Report Builder:** Create insightful custom reports without writing code, empowering data-driven decision-making.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting (Frappe Cloud)

For the easiest deployment, consider [Frappe Cloud](https://frappecloud.com), a user-friendly, open-source platform designed for hosting Frappe applications. It handles installation, upgrades, and maintenance, so you can focus on development.

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

**Prerequisites:** Docker, docker-compose, git. Refer to [Docker Documentation](https://docs.docker.com) for Docker setup details.

To get started:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a few minutes, your site should be accessible at `http://localhost:8080`. Use the following default login credentials:

*   Username: `Administrator`
*   Password: `admin`

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The easiest way is with the install script for bench, which installs all dependencies, including MariaDB. See [https://github.com/frappe/bench](https://github.com/frappe/bench) for more details. This script creates passwords for the Frappe Administrator user, the MariaDB root user, and the frappe user and saves the passwords to `~/frappe_passwords.txt`.

### Local

To set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, run the following commands:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open the URL `http://frappe.localhost:8000/app` in your browser, and you should see the app running.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext through community and maintainer-created courses.
2.  [Official documentation](https://docs.frappe.io/framework) - Extensive documentation for Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/) - Connect with the Frappe community and service providers.
4.  [buildwithhussain.com](https://buildwithhussain.com) - Watch Frappe Framework in action, building real-world web applications.

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