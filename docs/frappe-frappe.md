<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
    <h1>Frappe Framework: Low-Code Web Development for Real-World Applications</h1>
</div>

<div align="center">
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
    <a href="https://github.com/frappe/frappe">View on GitHub</a>
</div>

## About Frappe Framework

**Frappe Framework is a powerful, open-source, low-code web application framework, built with Python and JavaScript, enabling developers to build complex, real-world applications rapidly.** Inspired by the Semantic Web, Frappe allows for easy definition of metadata, making building applications that are consistent and extensible. Originally developed for ERPNext, Frappe is a full-stack solution, offering both front-end and back-end development capabilities.

## Key Features

*   **Full-Stack Development:** Build complete web applications with a single framework using Python (server-side) and JavaScript (client-side).
*   **Low-Code Approach:** Reduce development time with built-in features and a focus on metadata-driven development.
*   **Built-in Admin Interface:** Quickly manage application data and settings using a pre-built, customizable admin dashboard.
*   **Role-Based Permissions:** Implement robust user and role management to control access and permissions within your application.
*   **REST API:** Automatically generate RESTful APIs for all models, simplifying integration with other systems.
*   **Customizable Forms and Views:** Tailor forms and views using server-side scripting and client-side JavaScript.
*   **Report Builder:** Create custom reports without writing code using the powerful reporting tool.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for a simple and sophisticated open-source platform to host Frappe applications. It handles installation, upgrades, monitoring, maintenance, and support.

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

Prerequisites: Docker, Docker Compose, and Git.

Run the following commands:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a few minutes, your site should be accessible on your localhost port: 8080. Use the default login credentials to access the site:

*   Username: `Administrator`
*   Password: `admin`

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The Easy Way: Use the bench install script, which will install all dependencies (e.g., MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local

To set up the repository locally, follow these steps:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server.

    ```bash
    bench start
    ```

2.  In a separate terminal window, run the following commands:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open the URL `http://frappe.localhost:8000/app` in your browser; you should see the app running.

## Learning and Community

1.  [Frappe School](https://frappe.school): Learn Frappe Framework and ERPNext from courses by maintainers or the community.
2.  [Official Documentation](https://docs.frappe.io/framework): Extensive documentation for Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/): Engage with the community of Frappe Framework users and service providers.
4.  [buildwithhussain.com](https://buildwithhussain.com): Watch Frappe Framework used to build web applications.

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