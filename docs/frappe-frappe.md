<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>

 **Build powerful, real-world web applications quickly with Frappe, a low-code framework combining Python and JavaScript.**
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

## Frappe Framework: Low-Code Web Development

Frappe Framework is a full-stack, open-source web application framework designed for rapid development.  It leverages Python and MariaDB on the server-side and a tightly integrated client-side library.  Originally built for ERPNext, Frappe offers a powerful and efficient way to build complex web applications with a focus on semantics and extensibility. Learn more and contribute at the [original Frappe repository](https://github.com/frappe/frappe).

### Key Features:

*   **Full-Stack Development:**  Develop both front-end and back-end components within a single framework, streamlining your development process.
*   **Built-in Admin Interface:**  Save time and effort with a pre-built, customizable admin dashboard for easy data management.
*   **Role-Based Permissions:**  Implement robust user and role management to control access and permissions within your applications.
*   **REST API Generation:**  Automatically generate RESTful APIs for your models, enabling seamless integration with other services.
*   **Customizable Forms and Views:**  Flexibly customize forms and views using server-side scripting and client-side JavaScript.
*   **Report Builder:** Empower users to create custom reports without writing any code, facilitating data analysis and insights.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for a simple, user-friendly, and sophisticated open-source platform.

It manages installation, setup, upgrades, monitoring, maintenance, and support for your Frappe applications. This is a fully featured developer platform with the ability to manage and control multiple Frappe deployments.

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

*Prerequisites*: Docker, docker-compose, and git. Refer to [Docker Documentation](https://docs.docker.com) for detailed Docker setup instructions.

Run these commands:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a few minutes, the site should be accessible on your localhost at port 8080. Use the following default login credentials:

*   Username: Administrator
*   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

Install dependencies (e.g., MariaDB) easily with our bench install script. See [https://github.com/frappe/bench](https://github.com/frappe/bench) for more details.

The script creates new passwords for the "Administrator" user, the MariaDB root user, and the frappe user, displaying the passwords and saving them to `~/frappe_passwords.txt`.

### Local

Follow these steps to set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```
2.  In a separate terminal window, run these commands:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```
3.  Open the URL `http://frappe.localhost:8000/app` in your browser to see the running app.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext through courses by maintainers or the community.
2.  [Official documentation](https://docs.frappe.io/framework) - Extensive documentation for Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe Framework community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - Watch Frappe Framework in action, building world-class web applications.

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