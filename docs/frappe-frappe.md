<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
    <h1>Frappe Framework</h1>
    <p><b>Build powerful, real-world web applications quickly with the Frappe Framework, a low-code, full-stack solution built on Python and JavaScript.</b></p>
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="MIT License"></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Codecov"></a>
</div>

<div align="center">
    <img src=".github/hero-image.png" alt="Frappe Framework Hero Image" />
</div>

<div align="center">
    <a href="https://frappe.io/framework">Website</a>
    -
    <a href="https://docs.frappe.io/framework">Documentation</a>
    -
    <a href="https://github.com/frappe/frappe">View on GitHub</a>
</div>

## About Frappe Framework

Frappe Framework is a powerful, full-stack web application framework that leverages the efficiency of Python and MariaDB on the server-side, coupled with a tightly integrated client-side library for a seamless development experience. Designed initially for ERPNext, Frappe empowers developers to build robust and scalable applications with ease.  It emphasizes a data-driven approach, making your applications more consistent and extensible.

## Key Features

*   **Full-Stack Development:** Develop both front-end and back-end components using a single framework, streamlining your workflow.
*   **Low-Code Approach:** Reduce development time with built-in features like an Admin Interface and a REST API.
*   **Built-in Admin Interface:**  Provides a customizable admin dashboard for efficient data management.
*   **Role-Based Permissions:** Manage user access and permissions with a comprehensive system for enhanced security.
*   **REST API:** Automatically generate RESTful APIs for all your models, facilitating easy integration with other services.
*   **Customizable Forms and Views:** Tailor forms and views to your exact needs with server-side scripting and client-side JavaScript.
*   **Report Builder:** Create custom reports quickly without writing code using a powerful report builder.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started

### Production Setup

**Frappe Cloud (Managed Hosting):** Experience hassle-free deployment with [Frappe Cloud](https://frappecloud.com), a developer-friendly platform that handles installation, upgrades, monitoring, and maintenance for your Frappe applications.

<div align="center">
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

**Self-Hosting:**

#### Docker
Prerequisites: docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

Run following commands:

```
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a couple of minutes, site should be accessible on your localhost port: 8080. Use below default login credentials to access the site.
- Username: Administrator
- Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

### Development Setup

**Manual Install:**

1.  **Bench Setup:** Follow the [installation steps](https://docs.frappe.io/framework/user/en/installation) to setup bench and start the server:
    ```bash
    bench start
    ```

2.  **Create a New Site:** In a separate terminal window:
    ```bash
    bench new-site frappe.localhost
    ```

3.  **Access Your App:** Open `http://frappe.localhost:8000/app` in your browser to see your application running.

## Learning and Community

*   [Frappe School](https://frappe.school): Courses to learn Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.frappe.io/framework): Comprehensive documentation for Frappe Framework.
*   [Discussion Forum](https://discuss.frappe.io/): Engage with the Frappe Framework community.
*   [buildwithhussain.com](https://buildwithhussain.com): Watch Frappe Framework being used in real-world web app development.

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