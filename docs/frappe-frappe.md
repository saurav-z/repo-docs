<div align="center">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework</h1>
    <p><b>Build powerful, data-driven web applications quickly with Frappe Framework, a low-code platform built with Python and JavaScript.</b></p>
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

## Frappe Framework: The Low-Code Powerhouse for Web Applications

Frappe Framework is a full-stack, open-source web application framework that streamlines the development of complex, data-intensive web applications. Utilizing Python and MariaDB on the server-side, and a tightly integrated client-side library, Frappe empowers developers to build robust applications with significantly reduced coding effort.  Inspired by the Semantic Web, Frappe focuses on the meaning and relationships of your data, ensuring consistency, extensibility, and ease of development.  Learn more and contribute to the project on [GitHub](https://github.com/frappe/frappe).

### Key Features of Frappe Framework:

*   **Full-Stack Development**:  Develop complete applications, front-end and back-end, within a single framework.
*   **Built-in Admin Interface**:  Reduce development time with a pre-built, customizable admin dashboard for data management.
*   **Role-Based Permissions**: Implement granular access control with a comprehensive user and role management system.
*   **REST API**: Automatically generate a RESTful API for all models, enabling seamless integration with other systems.
*   **Customizable Forms and Views**:  Tailor forms and views using server-side scripting and client-side JavaScript for a personalized user experience.
*   **Report Builder**: Create custom reports easily with a powerful reporting tool, eliminating the need for extensive coding.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started with Frappe

### Production Setup

#### Managed Hosting with Frappe Cloud

For a hassle-free deployment experience, consider [Frappe Cloud](https://frappecloud.com). This user-friendly platform provides a managed hosting solution for Frappe applications, handling installation, updates, monitoring, and support.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

#### Self Hosting

**Prerequisites:** `docker`, `docker-compose`, `git`. Refer to the [Docker Documentation](https://docs.docker.com) for installation instructions.

**Docker Setup:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the docker-compose command:
    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, access your site at `http://localhost:8080`.  Use the default login credentials:

*   Username: `Administrator`
*   Password: `admin`

**ARM-Based Docker Setup:**  Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for instructions.

### Development Setup

#### Manual Install

The Easy Way: Utilize the bench install script for automatic dependency installation, including MariaDB. See [Bench Documentation](https://github.com/frappe/bench) for details.

*   The script will create new passwords for the Frappe "Administrator" user, the MariaDB root user, and the frappe user. Passwords are displayed and saved to `~/frappe_passwords.txt`.

#### Local Setup:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:
    ```bash
    bench start
    ```
2.  In a separate terminal window, run these commands:
    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```
3.  Open `http://frappe.localhost:8000/app` in your browser. You should see the running application.

## Learning and Community Resources

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext from community-contributed courses.
2.  [Official Documentation](https://docs.frappe.io/framework) - Comprehensive documentation for Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - Watch Frappe Framework used to build real-world applications.

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