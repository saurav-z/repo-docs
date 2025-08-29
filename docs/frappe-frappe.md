<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework: Low-Code Web Development for Real-World Applications</h1>
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
    <a href="https://github.com/frappe/frappe"><b>View on GitHub</b></a>
</div>

## Frappe Framework: Build Powerful Web Apps with Ease

Frappe Framework is a full-stack, low-code web application framework built with Python and JavaScript, empowering developers to rapidly build robust and scalable applications. Inspired by the Semantic Web, Frappe allows you to define your application's data and structure, making it easy to build complex apps that are consistent and extensible.

### Key Features

*   **Full-Stack Development:** Develop both front-end and back-end using Python and JavaScript within a single framework.
*   **Low-Code Approach:** Reduce development time with built-in features like an admin interface, form builders, and a report builder.
*   **Admin Interface:** A pre-built, customizable admin dashboard simplifies data management and application configuration.
*   **Role-Based Permissions:** Implement granular user and role management to control access to your application's data and features.
*   **REST API Generation:** Automatic REST API generation for all your models, facilitating integration with other systems.
*   **Customizable Forms and Views:** Build custom forms and views with server-side scripting and client-side JavaScript.
*   **Report Builder:** Create custom reports without writing any code, enabling data-driven insights.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting - Frappe Cloud

Frappe Cloud offers a simple and user-friendly platform to host your Frappe applications. It handles installation, setup, upgrades, monitoring, maintenance, and support.

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

**Prerequisites:** Docker, docker-compose, git. Refer to [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

**Steps:**

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run the Docker Compose file:

    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your site should be accessible on `localhost:8080`. Use the default login credentials:

*   **Username:** Administrator
*   **Password:** admin

Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

1.  Follow the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) for bench setup.
2.  Start the server:

    ```bash
    bench start
    ```

3.  In a separate terminal, create a new site:

    ```bash
    bench new-site frappe.localhost
    ```

4.  Open `http://frappe.localhost:8000/app` in your browser.

## Learning and Community

*   [Frappe School](https://frappe.school): Learn Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.frappe.io/framework): Comprehensive documentation.
*   [Discussion Forum](https://discuss.frappe.io/): Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com): See Frappe Framework in action.

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