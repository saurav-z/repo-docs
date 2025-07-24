<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework</h1>
    **Build powerful, semantic web applications quickly with this low-code Python and JavaScript framework.**
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

## Frappe Framework: Build Real-World Web Apps Faster

Frappe Framework is a full-stack, low-code web application framework built with Python and JavaScript, designed for rapidly building complex and scalable web applications. Originally created for ERPNext, Frappe offers a unique, semantic approach to web development.

[View the original repository](https://github.com/frappe/frappe).

### Key Features

*   **Full-Stack Development:** Develop both front-end and back-end within a single framework, streamlining the entire development process.
*   **Low-Code Approach:** Reduces development time by offering built-in features and a user-friendly interface.
*   **Admin Interface:** Provides a built-in and customizable admin dashboard, simplifying data management and reducing development effort.
*   **Role-Based Permissions:** Implement robust user and role management for granular control over application access and permissions.
*   **REST API Generation:** Automatically generates a RESTful API for all models, enabling seamless integration with other systems and services.
*   **Customizable Forms and Views:** Utilize server-side scripting and client-side JavaScript for flexible form and view customization.
*   **Report Builder:** Empower users to create custom reports without writing code, providing valuable insights.
*   **Semantic-Driven:** Create applications based on the meaning of the data, leading to more consistent and extensible applications.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting: Frappe Cloud

For a hassle-free experience, consider [Frappe Cloud](https://frappecloud.com), a developer-friendly platform for hosting Frappe applications. It handles installation, upgrades, monitoring, and support, allowing you to focus on building your application.

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

**Prerequisites:** Docker, Docker Compose, and Git. Refer to [Docker Documentation](https://docs.docker.com) for setup details.

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

After a few minutes, your site will be accessible on `localhost:8080`. Use the following default credentials to log in:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Installation

Use the install script for bench, which handles all dependencies like MariaDB; more info at:  https://github.com/frappe/bench.

The script will create new passwords for the "Administrator" user, MariaDB root, and the frappe user, saving them to `~/frappe_passwords.txt`.

### Local Setup

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, run:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open `http://frappe.localhost:8000/app` in your browser to see your running app.

## Learning and Community Resources

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext through courses.
2.  [Official documentation](https://docs.frappe.io/framework) - Comprehensive documentation for Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe Framework community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - See Frappe Framework in action, building real-world web apps.

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