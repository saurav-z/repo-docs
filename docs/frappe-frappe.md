<div align="center" markdown="1">
  <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
  <h1>Frappe Framework</h1>
  <p><strong>Build powerful, real-world web applications quickly with Frappe, a low-code framework built with Python and JavaScript.</strong></p>
</div>

<div align="center">
  <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="MIT License"></a>
  <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Codecov"></a>
</div>
<div align="center">
  <img src=".github/hero-image.png" alt="Hero Image" />
</div>
<div align="center">
    <a href="https://frappe.io/framework">Website</a> |
    <a href="https://docs.frappe.io/framework">Documentation</a> |
    <a href="https://github.com/frappe/frappe">View on GitHub</a>
</div>

## What is Frappe Framework?

Frappe Framework is a full-stack, open-source web application framework that accelerates web development. It leverages Python and MariaDB on the server-side, with a tightly integrated client-side library.  Inspired by the Semantic Web, Frappe allows developers to define the *meaning* of data, leading to more consistent, extensible, and powerful applications. Initially built for ERPNext, Frappe is a versatile tool for building a wide range of applications.

### Key Features

*   **Full-Stack Development**: Develop both front-end and back-end using a unified framework.
*   **Built-in Admin Interface**: Get a customizable admin dashboard for managing your application's data out-of-the-box.
*   **Role-Based Permissions**: Implement robust user and role management to control access within your application.
*   **REST API Generation**: Automatically generate RESTful APIs for your models, facilitating seamless integration.
*   **Customizable Forms and Views**: Tailor forms and views using server-side scripting and client-side JavaScript for a unique user experience.
*   **Report Builder**: Empower users to create custom reports without needing to write code.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started

### Production Setup

#### Managed Hosting - Frappe Cloud

For ease of use and peace of mind, consider [Frappe Cloud](https://frappecloud.com). This platform offers a streamlined approach to hosting Frappe applications, taking care of installation, upgrades, monitoring, maintenance, and support. It's a fully featured developer platform capable of managing multiple Frappe deployments.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

#### Self Hosting - Docker

**Prerequisites**:  Docker, docker-compose, git. Refer to [Docker Documentation](https://docs.docker.com) for Docker setup details.

**Steps**:

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the Docker Compose command:
    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your site should be accessible on `localhost:8080`. Use the following default login credentials:

*   **Username**: Administrator
*   **Password**: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for instructions on setting up a Docker environment on ARM64 architecture.

### Development Setup

#### Manual Install

The recommended approach for a quick start is to use the installation script for bench, which will install all dependencies, including MariaDB.  See the [Frappe bench documentation](https://github.com/frappe/bench) for more details.

The script creates new passwords for the "Administrator" user, the MariaDB root user, and the frappe user. These passwords are displayed and saved to `~/frappe_passwords.txt`.

#### Local

To set up a local development environment:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```
2.  In a separate terminal, create a new site:

    ```bash
    bench new-site frappe.localhost
    ```
3.  Open the URL `http://frappe.localhost:8000/app` in your browser; you should see the application running.

## Learning and Community

*   [Frappe School](https://frappe.school) - Courses on Frappe Framework and ERPNext.
*   [Official documentation](https://docs.frappe.io/framework) - Comprehensive documentation.
*   [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com) - Learn by watching Frappe Framework in action.

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