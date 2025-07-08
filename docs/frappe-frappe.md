<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>
    <p><b>Build powerful, semantic web applications with ease using the open-source Frappe Framework.</b></p>
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

## About Frappe Framework

Frappe Framework is a full-stack, low-code web application framework that uses Python and MariaDB on the server-side and a tightly integrated client-side library. It simplifies web development by providing a robust set of features and tools, ideal for building real-world applications.  Inspired by the Semantic Web, Frappe focuses on defining the *meaning* of your data, leading to more consistent and extensible applications.  Learn more and contribute at the [original repository](https://github.com/frappe/frappe).

## Key Features

*   **Full-Stack Development:** Develop complete web applications with both front-end and back-end capabilities within a single framework.
*   **Built-in Admin Interface:**  Get a ready-made, customizable admin dashboard to streamline data management, saving you time and effort.
*   **Role-Based Permissions:** Implement comprehensive user and role management for secure and controlled application access.
*   **REST API Generation:**  Automatically generate RESTful APIs for your models, enabling easy integration with other systems.
*   **Customizable Forms and Views:** Tailor forms and views using server-side scripting and client-side JavaScript to match your specific needs.
*   **Report Builder:** Empower users to create custom reports without writing any code, gaining valuable insights.
*   **Low Code:** Reduce the amount of boilerplate code needed to build complex applications.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Frappe Cloud offers a simple, user-friendly, and sophisticated open-source platform to host Frappe applications with ease.

It handles installation, upgrades, monitoring, maintenance, and support for your Frappe deployments. It's a fully featured developer platform with the ability to manage and control multiple Frappe deployments.

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

Prerequisites: docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for details on Docker setup.

To get started:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Your site should be accessible on localhost port 8080 after a few minutes. Use the following credentials:
- Username: Administrator
- Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The Easy Way: Our install script for bench will install all dependencies (e.g., MariaDB). See [bench documentation](https://github.com/frappe/bench) for more details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

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
3.  Open the URL `http://frappe.localhost:8000/app` in your browser; you should see the app running.

## Resources for Learning and Community

1.  [Frappe School](https://frappe.school): Courses to learn Frappe Framework and ERPNext from the maintainers and community.
2.  [Official Documentation](https://docs.frappe.io/framework): Comprehensive documentation for Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/): Engage with other Frappe Framework users and service providers.
4.  [buildwithhussain.com](https://buildwithhussain.com): See Frappe Framework in action, building world-class web apps.

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