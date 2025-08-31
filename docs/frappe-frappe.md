<!--  Frappe Framework - Low-Code Web Framework for Rapid Application Development -->
<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework: Build Powerful Web Apps with Ease</h1>
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

## Frappe Framework: The Low-Code Solution for Web App Development

**Frappe Framework** is a full-stack, low-code web application framework built with Python and JavaScript, perfect for building robust, real-world applications quickly and efficiently. Designed for rapid development, it provides a comprehensive suite of features to streamline your development process.  Explore the original repository on [GitHub](https://github.com/frappe/frappe).

### Key Features:

*   **Full-Stack Framework:** Develop both front-end and back-end components seamlessly within a single framework.
*   **Low-Code Development:**  Reduce development time with pre-built components and features.
*   **Built-in Admin Interface:**  Manage application data effortlessly with a customizable admin dashboard.
*   **Role-Based Permissions:** Implement robust security with granular user and role management.
*   **REST API:** Automatically generates RESTful APIs for easy integration with other systems.
*   **Customizable Forms and Views:** Design flexible forms and views using server-side scripting and client-side JavaScript.
*   **Report Builder:** Create custom reports without writing code, empowering users to analyze data effectively.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started

### Production Setup

*   **Frappe Cloud:** Consider [Frappe Cloud](https://frappecloud.com) for hassle-free hosting, which handles installation, upgrades, and maintenance.

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

**Prerequisites:** Docker, Docker Compose, Git.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  **Run with Docker Compose:**
    ```bash
    docker compose -f pwd.yml up -d
    ```

    Your site should be accessible at `http://localhost:8080`.
    Use "Administrator" / "admin" to login.

    For ARM-based Docker setup, refer to the [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) documentation.

## Development Setup

### Manual Installation
The installation instructions for bench are available in the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) documentation.

1.  **Setup bench:** by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation)
2.  **Start the Server:**
    ```bash
    bench start
    ```
3.  **In a separate terminal create a new site:**
    ```bash
    bench new-site frappe.localhost
    ```
4.  Open `http://frappe.localhost:8000/app` in your browser to see the running app.

## Resources and Community

*   **Frappe School:** [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext from the various courses by the maintainers or from the community.
*   **Documentation:** [Official documentation](https://docs.frappe.io/framework)
*   **Discussion Forum:** [Discussion Forum](https://discuss.frappe.io/)
*   **Tutorials:** [buildwithhussain.com](https://buildwithhussain.com) - Watch Frappe Framework being used in the wild to build world-class web apps.

## Contributing

*   **Issue Guidelines:** [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   **Report Security Vulnerabilities:** [Report Security Vulnerabilities](https://frappe.io/security)
*   **Pull Request Requirements:** [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   **Translations:** [Translations](https://crowdin.com/project/frappe)

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