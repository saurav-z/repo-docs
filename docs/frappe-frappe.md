<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>

 **Build robust, real-world web applications quickly with the Frappe Framework, a low-code solution built on Python and JavaScript.**
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

## Frappe Framework: The Low-Code Powerhouse

Frappe Framework is a full-stack, open-source web application framework that simplifies web development. It uses Python and MariaDB on the server-side, integrated with a powerful client-side library. Inspired by the Semantic Web, Frappe focuses on metadata-driven development, making complex applications easier to build, maintain, and extend.

**[Explore the original repository](https://github.com/frappe/frappe)**

### Key Features of Frappe Framework:

*   ✅ **Full-Stack Framework:** Develop both front-end and back-end with a unified approach, streamlining your development process.
*   ✅ **Built-in Admin Interface:** Quickly manage your application data with a pre-built, customizable admin dashboard.
*   ✅ **Role-Based Permissions:** Control user access and permissions precisely with a robust user and role management system.
*   ✅ **REST API:** Easily integrate with other systems through automatically generated RESTful APIs for all your models.
*   ✅ **Customizable Forms and Views:** Tailor forms and views to your specific needs with server-side scripting and client-side JavaScript.
*   ✅ **Report Builder:** Create custom reports effortlessly using the powerful, code-free reporting tool.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Simplify your deployments with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. Frappe Cloud handles installation, upgrades, monitoring, maintenance, and support.

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

**Prerequisites:** docker, docker-compose, git.  Refer to [Docker Documentation](https://docs.docker.com) for setup.

**Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the Docker Compose command:
    ```bash
    docker compose -f pwd.yml up -d
    ```

Your site should be accessible on `localhost:8080` after a few minutes.
Use the default login credentials below to access:
    *   Username: Administrator
    *   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The easiest way is using the install script for bench, which installs dependencies like MariaDB.  See [Frappe Bench](https://github.com/frappe/bench) for details.

The script will generate passwords for the Administrator user, MariaDB root user, and frappe user (passwords are saved in `~/frappe_passwords.txt`).

### Local Setup

To set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:
    ```bash
    bench start
    ```
2.  In a separate terminal:
    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```
3.  Open `http://frappe.localhost:8000/app` in your browser to see the app running.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext through courses.
2.  [Official documentation](https://docs.frappe.io/framework) - Extensive documentation.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - Watch Frappe Framework in action.

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