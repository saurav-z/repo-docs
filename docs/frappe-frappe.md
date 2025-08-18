<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework: Low-Code Web Development</h1>
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
    -
    <a href="https://github.com/frappe/frappe">View on GitHub</a>
</div>

## Frappe Framework: Build Powerful Web Apps with Ease

Frappe Framework is a full-stack, low-code web application framework built with Python and JavaScript, enabling developers to rapidly build real-world applications. Designed for efficiency and scalability, Frappe simplifies complex web development tasks, allowing you to focus on your application's core functionality.

## Key Features

*   **Full-Stack Development:** Develop both front-end and back-end applications with a single, integrated framework.
*   **Low-Code Approach:** Reduce development time with built-in features, automated API generation, and customizable components.
*   **Built-in Admin Interface:** Manage your application data effortlessly with a pre-built, customizable admin dashboard.
*   **Role-Based Permissions:** Implement robust user and role management to control access and security.
*   **REST API Generation:** Automatically generate RESTful APIs for all models, facilitating seamless integration.
*   **Customizable Forms and Views:** Tailor forms and views with server-side scripting and client-side JavaScript for a personalized user experience.
*   **Report Builder:** Create custom reports without writing code, empowering users to analyze data effectively.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for hassle-free Frappe application hosting. It offers:

*   Simplified installation, setup, and upgrades.
*   Monitoring, maintenance, and dedicated support.
*   A comprehensive developer platform for managing deployments.

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

Prerequisites: Docker, Docker Compose, Git. Refer to [Docker Documentation](https://docs.docker.com) for setup details.

1.  Clone the repository:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Your site will be accessible on localhost:8080 after a couple of minutes. Use "Administrator" / "admin" credentials to log in.

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The easiest way is to use the install script for bench, which installs dependencies (MariaDB): See https://github.com/frappe/bench for more details.

Passwords are created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script saves them to `~/frappe_passwords.txt`).

### Local Setup

1.  Set up bench following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

```bash
bench start
```

2.  In a separate terminal:

```bash
# Create a new site
bench new-site frappe.localhost
```

3.  Access the app at `http://frappe.localhost:8000/app`.

## Learning and Community

*   [Frappe School](https://frappe.school): Learn Frappe Framework and ERPNext.
*   [Official documentation](https://docs.frappe.io/framework): Comprehensive documentation.
*   [Discussion Forum](https://discuss.frappe.io/): Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com): Real-world Frappe app development examples.

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