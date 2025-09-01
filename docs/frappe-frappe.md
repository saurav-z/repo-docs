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

**Frappe Framework**, a low-code web framework, empowers developers to build real-world applications quickly and efficiently using Python and JavaScript.  Explore the original repository [here](https://github.com/frappe/frappe).

### Key Features

*   **Full-Stack Development:**  Develop both the front-end and back-end of your applications with a single framework.
*   **Built-in Admin Interface:**  Save time and effort with a customizable admin dashboard.
*   **Role-Based Permissions:**  Manage user access and control data security effectively.
*   **REST API Generation:**  Integrate your application seamlessly with other services through automatically generated APIs.
*   **Customizable Forms and Views:**  Tailor forms and views to your specific needs using server-side scripting and client-side JavaScript.
*   **Report Builder:** Create custom reports without coding.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for hassle-free hosting of your Frappe applications. It handles installation, upgrades, monitoring, and support.

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

**Prerequisites:** docker, docker-compose, git. See [Docker Documentation](https://docs.docker.com) for Docker setup details.

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the Docker Compose file:
    ```bash
    docker compose -f pwd.yml up -d
    ```

Your site should be accessible on `localhost:8080`. Use the default login:

*   Username: Administrator
*   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

1.  Follow the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) to set up bench.
2.  Start the server: `bench start`
3.  In a separate terminal, create a new site: `bench new-site frappe.localhost`
4.  Open `http://frappe.localhost:8000/app` in your browser.

## Learning and Community

*   [Frappe School](https://frappe.school) - Courses and community knowledge base for learning Frappe Framework and ERPNext.
*   [Official documentation](https://docs.frappe.io/framework) - Comprehensive documentation for the Frappe Framework.
*   [Discussion Forum](https://discuss.frappe.io/) - Connect with other users and service providers.
*   [buildwithhussain.com](https://buildwithhussain.com) - See real-world Frappe Framework app development.

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