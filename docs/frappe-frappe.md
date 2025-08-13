<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework: Low-Code Web Development with Python and JavaScript</h1>
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

## Frappe Framework: Build Powerful Web Applications Faster

**Frappe Framework** is a full-stack, low-code web application framework that simplifies web development by providing a powerful backend built on Python and MariaDB, and a tightly integrated client-side library using JavaScript, enabling developers to build complex applications quickly.

[Visit the original repository](https://github.com/frappe/frappe)

## Key Features of Frappe Framework

*   **Full-Stack Framework:** Develop complete web applications with a single framework, covering both front-end and back-end development.
*   **Low-Code Approach:** Accelerate development with built-in features and a focus on metadata-driven development, minimizing the need for extensive coding.
*   **Built-in Admin Interface:** Manage application data efficiently with a pre-built, customizable admin dashboard, saving time and effort.
*   **Role-Based Permissions:** Implement robust user and role management to control access and permissions within your application.
*   **REST API:** Automatically generate RESTful APIs for all models, enabling seamless integration with other systems and services.
*   **Customizable Forms and Views:** Tailor forms and views to your specific needs using server-side scripting and client-side JavaScript.
*   **Report Builder:** Create custom reports without writing code using a powerful reporting tool.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for a user-friendly and sophisticated platform to host your Frappe applications. It handles installation, setup, upgrades, monitoring, maintenance, and support.

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

**Prerequisites**: docker, docker-compose, git. Refer to [Docker Documentation](https://docs.docker.com) for details.

**Steps:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a few minutes, your site should be accessible on `localhost:8080`.

**Login Credentials:**
*   Username: Administrator
*   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

Use the install script for bench: [Bench](https://github.com/frappe/bench).

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local

1.  **Setup Bench:** Follow the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  **Create a Site:** In a separate terminal:

    ```bash
    bench new-site frappe.localhost
    ```

3.  **Access your App:** Open `http://frappe.localhost:8000/app` in your browser.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Courses on Frappe Framework and ERPNext.
2.  [Official documentation](https://docs.frappe.io/framework) - Comprehensive Frappe Framework documentation.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe Framework community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - Learn by watching Frappe Framework being used in real-world projects.

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