<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework</h1>

    **Build powerful, semantic web applications rapidly with the Frappe Framework - a low-code powerhouse using Python and JavaScript.**
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

Frappe Framework is a full-stack web application framework, written in Python and JavaScript, that simplifies web development by providing a low-code environment.  Built upon semantic web principles, Frappe enables developers to build complex, data-driven applications efficiently. Originally created for ERPNext, Frappe is a robust framework for building real-world business applications.  Explore the source on [GitHub](https://github.com/frappe/frappe).

### Key Features

*   **Full-Stack Development:**  Develop both the front-end and back-end of your applications within a single framework, accelerating the development process.
*   **Built-in Admin Interface:**  Utilize a pre-built, customizable admin dashboard to manage application data with ease, reducing development time.
*   **Role-Based Permissions:**  Implement granular user and role management to control access and permissions within your application, enhancing security.
*   **REST API Generation:** Automatically generate RESTful APIs for all models, enabling seamless integration with other systems and services.
*   **Customizable Forms & Views:**  Tailor forms and views using server-side scripting and client-side JavaScript to meet your specific application requirements.
*   **Powerful Report Builder:**  Create custom reports without writing any code, empowering users with data-driven insights.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting: Frappe Cloud

For a simplified hosting experience, consider [Frappe Cloud](https://frappecloud.com).  This user-friendly platform offers a simple, open-source (Frappe's Docker files are [here](https://github.com/frappe/press)) solution for hosting Frappe applications, handling installation, upgrades, and maintenance.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Self-Hosting

#### Docker

**Prerequisites:**  Docker, Docker Compose, and Git installed. Refer to the [Docker Documentation](https://docs.docker.com) for setup instructions.

**Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    docker compose -f pwd.yml up -d
    ```
2.  Access your site on `localhost:8080` using the default credentials:
    *   Username: `Administrator`
    *   Password: `admin`

   See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The easiest method involves utilizing the bench install script, which handles all dependencies, including MariaDB. See [Frappe Bench documentation](https://github.com/frappe/bench) for details.

> Note: The script creates new passwords for the Frappe "Administrator" user, MariaDB root, and the Frappe user, displaying them and saving them to `~/frappe_passwords.txt`.

### Local Setup

Follow these steps to set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server
    ```bash
    bench start
    ```
2.  In a separate terminal, run:
    ```bash
    bench new-site frappe.localhost
    ```
3.  Open `http://frappe.localhost:8000/app` in your browser.  You should see the application running.

## Learning and Community Resources

1.  [Frappe School](https://frappe.school) - Learn the Frappe Framework and ERPNext through courses from the maintainers and community.
2.  [Official Documentation](https://docs.frappe.io/framework) - Comprehensive documentation for the Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe Framework community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - Watch Frappe Framework in action, building real-world web applications.

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