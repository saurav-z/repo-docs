<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>

 **Build powerful, real-world web applications faster with Frappe, a low-code framework powered by Python and JavaScript.**
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

## About Frappe Framework

Frappe Framework is a full-stack, open-source web application framework designed for rapid development. Built with Python and MariaDB on the backend and a tightly integrated JavaScript client-side library, Frappe simplifies the creation of complex, data-driven applications. Inspired by the principles of the Semantic Web, Frappe allows developers to focus on the meaning and relationships of data, leading to more consistent and extensible applications. It is the foundation for ERPNext and ideal for those who are ready to build robust web solutions.

## Key Features of Frappe Framework

*   **Full-Stack Development:** Develop both front-end and back-end components within a single framework.
*   **Low-Code Approach:** Minimize manual coding with built-in features and automated functionalities.
*   **Admin Interface:** Benefit from a pre-built, customizable admin dashboard for efficient data management.
*   **Role-Based Permissions:** Fine-grained control over user access and permissions for enhanced security.
*   **REST API:** Generate automated RESTful APIs for seamless integration with external systems.
*   **Customizable Forms and Views:** Utilize server-side scripting and client-side JavaScript for flexible form and view customization.
*   **Report Builder:** Create custom reports with a powerful reporting tool without writing any code.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for hassle-free Frappe application hosting. This user-friendly platform manages installation, upgrades, monitoring, and support, offering a fully-featured developer platform with control over multiple deployments.

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

**Prerequisites:** docker, docker-compose, git. Refer to [Docker Documentation](https://docs.docker.com) for Docker setup.

**Steps:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Your site should be accessible on `localhost:8080` after a few minutes. Use the following default login credentials:
- Username: Administrator
- Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based docker setup.

## Development Setup

### Manual Install

**The Easy Way:** Use the install script for bench which will install all dependencies (e.g. MariaDB). See [bench documentation](https://github.com/frappe/bench) for details.
New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local

Follow these steps to set up the repository locally:

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

## Learning and Community Resources

1.  [Frappe School](https://frappe.school) - Courses on Frappe Framework and ERPNext.
2.  [Official Documentation](https://docs.frappe.io/framework) - Comprehensive documentation.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - Examples of Frappe Framework in action.

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