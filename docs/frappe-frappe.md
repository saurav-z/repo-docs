<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>

 **Build powerful, semantic web applications faster with the Frappe Framework, a low-code platform built on Python and JavaScript.**
</div>

<div align="center">
	<a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg"></a>
	<a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj"/></a>
	<br>
	<a href="https://github.com/frappe/frappe">
	    <img src="https://img.shields.io/github/stars/frappe/frappe?style=social" alt="GitHub stars">
	</a>
</div>
<div align="center">
	<img src=".github/hero-image.png" alt="Hero Image" />
</div>
<div align="center">
    <a href="https://frappe.io/framework">Website</a>
    -
    <a href="https://docs.frappe.io/framework">Documentation</a>
</div>

## Frappe Framework: Your Full-Stack Web Application Powerhouse

Frappe Framework is a powerful, open-source, full-stack web application framework that simplifies building robust and scalable applications using Python on the server-side and a tightly integrated JavaScript client-side library. Inspired by semantic web principles, Frappe allows you to define the *meaning* of your data, leading to more consistent, extensible, and maintainable applications.

**[Learn More and Contribute on GitHub](https://github.com/frappe/frappe)**

### Key Features

*   **Full-Stack Development:** Build complete web applications with a unified framework for both front-end and back-end development.
*   **Low-Code Admin Interface:**  Quickly create and customize admin dashboards for efficient data management, saving development time.
*   **Role-Based Access Control:**  Implement granular user and role management to control access permissions and data security.
*   **RESTful API Generation:** Automatically generates a REST API for all models, making integration with other systems seamless.
*   **Customizable Forms and Views:**  Tailor forms and views to your specific needs using server-side scripting and client-side JavaScript for a personalized user experience.
*   **Powerful Report Builder:** Empower users to create custom reports without writing any code, facilitating data analysis and insights.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for a hassle-free, open-source platform to host your Frappe applications. It offers easy installation, setup, upgrades, monitoring, maintenance, and support for your deployments.

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

**Prerequisites**:  Docker, docker-compose, and git. Refer to [Docker Documentation](https://docs.docker.com) for detailed setup instructions.

**Installation**:

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run Docker Compose:
    ```bash
    docker compose -f pwd.yml up -d
    ```

After a couple of minutes, your site should be accessible on `localhost:8080`. Use the following credentials:

*   **Username:** Administrator
*   **Password:** admin

Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The easiest way is to use our install script for bench which installs all dependencies (e.g. MariaDB).  See [bench documentation](https://github.com/frappe/bench) for more details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local Development

Follow these steps to set up the repository locally:

1.  Install bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```
2.  In a separate terminal window:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```
3.  Open the URL `http://frappe.localhost:8000/app` in your browser. You should see the application running.

## Learning and Community

*   [Frappe School](https://frappe.school) - Courses for learning Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.frappe.io/framework) - Comprehensive documentation.
*   [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com) - See Frappe Framework in action building web apps.

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