<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>

 **Build powerful, real-world web applications quickly with the low-code Frappe Framework, combining Python and JavaScript.**
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

## Frappe Framework: Build Web Apps Faster with Low-Code

Frappe Framework is a full-stack, low-code web application framework that uses Python and MariaDB on the server side and a tightly integrated client-side library. It's designed for building robust, scalable applications, inspired by the semantic web to focus on the meaning of data rather than just its presentation. Originally developed for ERPNext, Frappe empowers developers to create complex applications with ease.

**[Explore the Original Repository](https://github.com/frappe/frappe)**

## Key Features

*   **Full-Stack Development:** Develop both front-end and back-end components within a single framework.
*   **Built-in Admin Interface:** Get a ready-to-use and customizable admin dashboard to manage your application's data efficiently.
*   **Role-Based Permissions:** Secure your application with a comprehensive user and role management system.
*   **REST API Generation:** Automatically generate RESTful APIs for seamless integration with other services.
*   **Customizable Forms and Views:** Tailor forms and views using server-side scripting and client-side JavaScript to match your specific requirements.
*   **Report Builder:** Easily create custom reports without writing code, using the powerful reporting tool.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting: Frappe Cloud

For ease of use, consider [Frappe Cloud](https://frappecloud.com), a user-friendly, open-source platform for hosting Frappe applications. It simplifies installation, upgrades, monitoring, and maintenance.

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

**Prerequisites:** Docker, Docker Compose, and Git. See [Docker Documentation](https://docs.docker.com) for setup details.

**Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    docker compose -f pwd.yml up -d
    ```
2.  Access your site on `localhost:8080` using the default credentials:
    *   Username: Administrator
    *   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The Easy Way: Use the install script for bench. See [Bench](https://github.com/frappe/bench) for more details.

### Local Setup

1.  Set up bench: Follow the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:
    ```bash
    bench start
    ```
2.  In a separate terminal:
    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```
3.  Open `http://frappe.localhost:8000/app` in your browser.

## Learning and Community

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext through courses.
*   [Official documentation](https://docs.frappe.io/framework) - Comprehensive documentation.
*   [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com) - See Frappe Framework in action building real-world apps.

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