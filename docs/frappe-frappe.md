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

## Frappe Framework: Build Powerful Web Applications Faster

Frappe Framework is a full-stack, low-code web application framework built with Python and JavaScript, allowing developers to rapidly build real-world applications. [Check out the original repository](https://github.com/frappe/frappe).

### Key Features

*   **Full-Stack Development:** Develop both front-end and back-end with a single framework.
*   **Built-in Admin Interface:**  Quickly manage data with a customizable admin dashboard.
*   **Role-Based Permissions:** Secure your application with robust user and role management.
*   **REST API Generation:**  Automatically generate RESTful APIs for easy integration.
*   **Customizable Forms & Views:** Tailor your application's interface with flexibility.
*   **Report Builder:** Create custom reports without writing any code.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started

### Production Setup

Consider using [Frappe Cloud](https://frappecloud.com) for a managed hosting solution.

Alternatively, self-host using Docker:

#### Docker Setup

**Prerequisites:** Docker, Docker Compose, Git.  Refer to [Docker Documentation](https://docs.docker.com) for setup.

**Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run Docker Compose:
    ```bash
    docker compose -f pwd.yml up -d
    ```
3.  Access your site at `http://localhost:8080`.
4.  Use default login credentials to access the site.
    *   Username: Administrator
    *   Password: admin

Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

### Development Setup

#### Manual Install
Follow the installation steps mentioned in the [installation document](https://docs.frappe.io/framework/user/en/installation) and start the server. 

#### Local

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

## Resources & Community

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext from the various courses by the maintainers or from the community.
*   [Official Documentation](https://docs.frappe.io/framework) - Comprehensive Frappe Framework documentation.
*   [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com) - See Frappe Framework in action.

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