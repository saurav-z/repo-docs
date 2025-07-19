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

## Frappe Framework: Build Powerful Web Apps Faster

**Frappe Framework** is a robust, low-code web framework empowering developers to build real-world applications efficiently, using Python and JavaScript.  Built for rapid development and scalability, Frappe Framework simplifies complex tasks and promotes consistent, extensible application design.  [Explore the original repository](https://github.com/frappe/frappe).

### Key Features

*   **Full-Stack Development:** Develop both front-end and back-end components seamlessly within a single framework, streamlining your workflow.
*   **Built-in Admin Interface:**  Reduce development time with a pre-built, customizable admin dashboard for efficient data management and application control.
*   **Role-Based Permissions:**  Implement granular user and role management to ensure secure access and control within your application.
*   **REST API Generation:** Automatically generate RESTful APIs for all models, enabling seamless integration with external systems and services.
*   **Customizable Forms and Views:**  Tailor forms and views to meet your specific needs using server-side scripting and client-side JavaScript for a personalized user experience.
*   **Report Builder:**  Empower users to create custom reports without writing code, enabling data-driven decision-making.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

For a hassle-free deployment, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform designed for hosting Frappe applications.  It provides comprehensive services, including installation, upgrades, monitoring, and support, making it ideal for developers.

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

**Prerequisites:** Docker, Docker Compose, and Git.

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

    Your site will be accessible on `localhost:8080`.  Use the default credentials:
    *   **Username:** Administrator
    *   **Password:** admin

Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The simplest option uses the bench install script.  This script sets up dependencies like MariaDB. See [bench](https://github.com/frappe/bench) for details.

The install script will generate passwords for the Frappe "Administrator" user, MariaDB root user, and the frappe user, saving them to `~/frappe_passwords.txt`.

### Local

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, create a new site:

    ```bash
    bench new-site frappe.localhost
    ```

3.  Open `http://frappe.localhost:8000/app` in your browser to see the running application.

## Learning and Community

1.  [Frappe School](https://frappe.school): Learn the framework and ERPNext through courses by the maintainers and community.
2.  [Official Documentation](https://docs.frappe.io/framework): Comprehensive documentation for Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/): Engage with the Frappe community.
4.  [buildwithhussain.com](https://buildwithhussain.com): See real-world Frappe Framework applications.

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