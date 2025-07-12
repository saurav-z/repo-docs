<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>
</div>

<div align="center">
    **Build powerful, real-world web applications quickly with Frappe, a low-code Python and JavaScript framework.**
</div>

<div align="center">
	<a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg"></a>
	<a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj"/></a>
	<a href="https://github.com/frappe/frappe">
		<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/frappe/frappe?style=social">
	</a>
</div>
<div align="center">
	<img src=".github/hero-image.png" alt="Hero Image" />
</div>
<div align="center">
    <a href="https://frappe.io/framework">Website</a>
    -
    <a href="https://docs.frappe.io/framework">Documentation</a>
    -
    <a href="https://github.com/frappe/frappe">View on GitHub</a>
</div>

## What is Frappe Framework?

Frappe Framework is a full-stack, low-code web application framework that streamlines web app development using Python and JavaScript. It provides a robust foundation for building complex, data-driven applications, particularly those with intricate business logic.  Built on a semantic understanding of data, Frappe makes application development more consistent and extensible, allowing you to focus on functionality rather than boilerplate code.

### Key Features of Frappe Framework:

*   **Full-Stack Development:**  Develop both front-end and back-end components using a single framework, boosting efficiency.
*   **Built-in Admin Interface:** Offers a pre-built, customizable admin dashboard to manage application data, saving valuable development time.
*   **Role-Based Permissions:**  Implement comprehensive user and role management for secure access control within your applications.
*   **REST API Generation:** Automatically generates RESTful APIs for all data models, facilitating seamless integration with other systems.
*   **Customizable Forms & Views:** Customize forms and views flexibly using server-side scripting and client-side JavaScript for a tailored user experience.
*   **Report Builder:** Create and generate custom reports without writing code using the powerful built-in reporting tool.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started

### Production Setup

Choose from the managed hosting of Frappe Cloud or self-hosting options for your Frappe applications.

### Managed Hosting: Frappe Cloud

[Frappe Cloud](https://frappecloud.com) provides a simple and sophisticated platform to host Frappe applications. It takes care of installation, upgrades, monitoring, and maintenance.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Self-Hosting:

Choose your hosting solution.

#### Docker

**Prerequisites:** docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

Run these commands to setup with Docker:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a few minutes, your site should be accessible on `localhost:8080`.
Default login credentials:
-   Username: Administrator
-   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based docker setup.

## Development Setup

### Manual Install

Follow the install script for bench to install all dependencies (e.g. MariaDB). See https://github.com/frappe/bench for details.
Passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local Setup

To set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server.
    ```bash
    bench start
    ```
2.  In a separate terminal window, run:
    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```
3.  Open `http://frappe.localhost:8000/app` in your browser.

## Learn and Connect

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext from courses by maintainers and the community.
*   [Official Documentation](https://docs.frappe.io/framework) - Detailed documentation for Frappe Framework.
*   [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe Framework community.
*   [buildwithhussain.com](https://buildwithhussain.com) - See Frappe Framework in action, building real-world web apps.

## Contribute

Help improve Frappe Framework!

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