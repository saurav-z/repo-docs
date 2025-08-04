<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>
</div>

**Build powerful, real-world web applications quickly with the Frappe Framework, a low-code platform built with Python and JavaScript.**

[<img src="https://img.shields.io/badge/License-MIT-success.svg" alt="License: MIT">](LICENSE)
[<img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Codecov">](https://codecov.io/gh/frappe/frappe)

<div align="center">
	<img src=".github/hero-image.png" alt="Hero Image" />
</div>

<div align="center">
    <a href="https://frappe.io/framework">Website</a>
    -
    <a href="https://docs.frappe.io/framework">Documentation</a>
</div>

## About Frappe Framework

Frappe Framework is a full-stack web application framework, using Python and MariaDB on the server side and a tightly integrated client-side library. It's designed for building complex, data-driven applications with an emphasis on a semantic approach, focusing on *what* your data *means* rather than just how it's displayed. Built initially for ERPNext, Frappe Framework offers a robust foundation for diverse web projects.

*   **[Explore the original repository on GitHub](https://github.com/frappe/frappe)**

### Key Features

*   **Full-Stack Development:**  Develop both front-end and back-end components within a unified framework, streamlining the development process.
*   **Built-in Admin Interface:**  Benefit from a pre-built, customizable admin dashboard to manage application data efficiently.
*   **Role-Based Permissions:** Implement fine-grained user and role management for secure and controlled access to your application's features.
*   **REST API Generation:** Automatically generate RESTful APIs for all your data models, facilitating seamless integration with other services.
*   **Customizable Forms and Views:** Create tailored user interfaces with flexible form and view customization using server-side scripting and client-side JavaScript.
*   **Report Builder:** Empower users with a powerful reporting tool to generate custom reports without coding.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for a hassle-free hosting solution, providing installation, upgrades, monitoring, and support.

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

**Prerequisites:** docker, docker-compose, git. See [Docker Documentation](https://docs.docker.com) for details.

Run the following commands:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Your site should be accessible on `localhost:8080` after a few minutes.  Use these default credentials:
- Username: Administrator
- Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The Easy Way: Install all dependencies (e.g., MariaDB) with our bench install script. See https://github.com/frappe/bench for details.

The script will generate new passwords for the "Administrator" user, MariaDB root, and frappe user, saving them to `~/frappe_passwords.txt`.

### Local

To set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```
2.  In a separate terminal window, run:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```
3.  Open `http://frappe.localhost:8000/app` in your browser.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Courses for learning Frappe Framework and ERPNext.
2.  [Official documentation](https://docs.frappe.io/framework) - Extensive documentation.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - Watch real-world Frappe Framework application builds.

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