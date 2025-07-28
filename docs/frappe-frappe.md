<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>

 **Build powerful and scalable web applications with ease using the Frappe Framework, a low-code solution built on Python and JavaScript.**
</div>

<div align="center">
	<a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg"></a>
	<a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj"/></a>
    <br>
    <a href="https://github.com/frappe/frappe">View on GitHub</a>
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

Frappe Framework is a full-stack web application framework that simplifies development by using Python and MariaDB on the server and a tightly integrated client-side library. Originally built to power ERPNext, Frappe offers a rapid application development experience by focusing on the semantics of your data, leading to more consistent and extensible applications.

### Key Features

*   ✅ **Full-Stack Development:** Build complete web applications with a single framework, encompassing both front-end and back-end development using Python and JavaScript.
*   ✅ **Built-in Admin Interface:** Leverage a pre-built, customizable admin dashboard to streamline data management and reduce development time.
*   ✅ **Role-Based Permissions:** Implement robust user and role management to control access and permissions within your applications.
*   ✅ **REST API Generation:** Automatically generate RESTful APIs for all your models, facilitating easy integration with external systems and services.
*   ✅ **Customizable Forms and Views:** Create flexible forms and views with server-side scripting and client-side JavaScript to tailor your user interfaces.
*   ✅ **Report Builder:** Design custom reports without writing any code, empowering users to analyze data effectively.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Simplify your Frappe deployments with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. Frappe Cloud handles installation, upgrades, monitoring, maintenance, and support, offering a fully featured developer platform to manage multiple deployments.

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

**Prerequisites:** docker, docker-compose, git. Refer to [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

**Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run the Docker Compose setup:
    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your site will be accessible on your localhost at port 8080. Use the following default credentials to access the site:
-   Username: Administrator
-   Password: admin

For ARM-based Docker setups, see [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Install

The easiest way to install dependencies (e.g., MariaDB) is using our bench install script, which is part of the [Frappe Bench](https://github.com/frappe/bench) tool. New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local

To set up the repository locally, follow these steps:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:
    ```bash
    bench start
    ```

2.  In a separate terminal window, run the following commands:
    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open the URL `http://frappe.localhost:8000/app` in your browser; you should see the application running.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext with courses from the maintainers and the community.
2.  [Official documentation](https://docs.frappe.io/framework) - Explore comprehensive documentation for Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe Framework community and find solutions.
4.  [buildwithhussain.com](https://buildwithhussain.com) - Watch Frappe Framework being used to build web applications.

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