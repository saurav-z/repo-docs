<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
    <h1>Frappe Framework</h1>

    **Build powerful, real-world web applications quickly with the Frappe Framework, a low-code web framework built on Python and JavaScript.**
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="License: MIT"></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Code Coverage"/></a>
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

## Frappe Framework: The Low-Code Powerhouse for Web Application Development

Frappe Framework is a full-stack, open-source web application framework designed to streamline the development of complex applications.  It leverages Python and MariaDB on the server-side, with a tightly integrated client-side library, making it a robust choice for building modern web applications. Originally designed for ERPNext, Frappe is now a versatile tool for a wide range of projects.

### Key Features

*   **Full-Stack Development:** Develop both front-end and back-end with a single, unified framework, boosting efficiency and maintainability.
*   **Low-Code Approach:** Minimize the amount of code you need to write with Frappe's built-in features and intuitive design.
*   **Built-in Admin Interface:** Get up and running fast with a pre-built, customizable admin dashboard to manage your application data effortlessly.
*   **Role-Based Permissions:** Secure your application with a comprehensive user and role management system, ensuring granular control over access and permissions.
*   **REST API Generation:** Automatically generate RESTful APIs for all your models, enabling seamless integration with other systems and services.
*   **Customizable Forms and Views:** Tailor forms and views to your exact requirements using server-side scripting and client-side JavaScript.
*   **Report Builder:** Empower users to create custom reports without needing to write any code, facilitating data-driven decision-making.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Simplify your deployments with [Frappe Cloud](https://frappecloud.com), a user-friendly, open-source platform.

It provides installation, setup, upgrades, monitoring, maintenance, and support of your Frappe deployments.

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

*   Username: Administrator
*   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The Easy Way: our install script for bench will install all dependencies (e.g. MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

To setup the repository locally follow the steps mentioned below:

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

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext from the various courses by the maintainers or from the community.
2.  [Official documentation](https://docs.frappe.io/framework) - Extensive documentation for Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with community of Frappe Framework users and service providers.
4.  [buildwithhussain.com](https://buildwithhussain.com) - Watch Frappe Framework being used in the wild to build world-class web apps.

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