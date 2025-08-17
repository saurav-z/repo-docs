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
</div>

## Frappe Framework: Build Powerful Web Applications with Ease

Frappe Framework is a robust, low-code web application framework using Python and JavaScript, designed for building complex and scalable applications. Inspired by semantic web principles, it emphasizes data meaning for consistent and extensible applications, perfect for developers seeking a powerful and efficient development experience. Learn more and contribute on the [Frappe Framework GitHub repository](https://github.com/frappe/frappe).

### Key Features:

*   **Full-Stack Development:** Develop both front-end and back-end components within a single framework, simplifying the development process.
*   **Built-in Admin Interface:** Get a head start with a customizable admin dashboard for streamlined data management, saving valuable development time.
*   **Role-Based Permissions:** Control user access with a comprehensive role management system, ensuring data security and access control.
*   **REST API:** Generate RESTful APIs automatically for easy integration with other systems and services.
*   **Customizable Forms and Views:** Leverage server-side scripting and client-side JavaScript for flexible customization of forms and views.
*   **Report Builder:** Create custom reports without any coding, enabling data visualization and analysis for informed decision-making.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Simplify your deployments with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. Frappe Cloud handles installation, upgrades, monitoring, and support.

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

Prerequisites: docker, docker-compose, git.

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run the Docker Compose file:
    ```bash
    docker compose -f pwd.yml up -d
    ```

Access your application at `localhost:8080` using the default credentials:
*   Username: Administrator
*   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The easy way uses the install script for bench, which installs all dependencies. See [Bench Installation](https://github.com/frappe/bench) for details.

New passwords are created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local Setup

1.  Set up bench following the [installation steps](https://docs.frappe.io/framework/user/en/installation) and start the server.
    ```bash
    bench start
    ```

2.  In a separate terminal, run:
    ```bash
    bench new-site frappe.localhost
    ```

3.  Open `http://frappe.localhost:8000/app` in your browser.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Courses on Frappe Framework and ERPNext.
2.  [Official Documentation](https://docs.frappe.io/framework) - Comprehensive framework documentation.
3.  [Discussion Forum](https://discuss.frappe.io/) - Community forum for support and discussion.
4.  [buildwithhussain.com](https://buildwithhussain.com) - See Frappe in action.

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