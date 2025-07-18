<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework: The Low-Code Powerhouse</h1>
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

## Frappe Framework: Build Powerful Web Applications Faster

Frappe Framework is a **full-stack, low-code web framework** built on Python and JavaScript, designed for real-world applications.  It offers a rapid development environment, letting you build complex web apps efficiently.  Inspired by the semantic web, Frappe focuses on the underlying meaning of your data, making your applications more consistent, extensible, and easier to maintain.

**[Visit the original repository on GitHub](https://github.com/frappe/frappe)**.

### Key Features:

*   **Full-Stack Development:** Develop both front-end and back-end using a single framework.
*   **Built-in Admin Interface:** Get a customizable admin dashboard to easily manage your application data.
*   **Role-Based Permissions:**  Implement robust user and role management for secure access control.
*   **Automated REST API:**  Integrate your application with other systems effortlessly through automatically generated REST APIs.
*   **Customizable Forms and Views:**  Use server-side scripting and client-side JavaScript for flexible form and view customization.
*   **Powerful Report Builder:**  Create custom reports without needing to write any code.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Simplify your deployments with [Frappe Cloud](https://frappecloud.com), an open-source platform that handles installation, upgrades, and maintenance.

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

**Prerequisites:** docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  **Run Docker Compose:**

    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your site should be accessible on your localhost at port 8080. Use the following default login credentials:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

1.  **Bench Setup:** Follow the [installation steps](https://docs.frappe.io/framework/user/en/installation) for bench and start the server:

    ```bash
    bench start
    ```

2.  **Create a New Site:**  In a separate terminal, run:

    ```bash
    bench new-site frappe.localhost
    ```

3.  **Access the App:** Open `http://frappe.localhost:8000/app` in your browser.

## Learning and Community

*   [Frappe School](https://frappe.school) - Courses on Frappe Framework and ERPNext.
*   [Official documentation](https://docs.frappe.io/framework) - Comprehensive documentation.
*   [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com) - Watch Frappe Framework in action.

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