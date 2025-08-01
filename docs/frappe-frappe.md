<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>

 **Build powerful, real-world web applications faster with the Frappe Framework, a low-code, full-stack solution.**
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
	- <a href="https://github.com/frappe/frappe">View on GitHub</a>
</div>

## About Frappe Framework

Frappe Framework is a robust, full-stack web application framework built on Python and JavaScript. It provides a powerful and flexible platform for developing a wide range of web applications, including the popular ERPNext system.  Frappe emphasizes a semantic-driven approach, allowing you to define your application's data and logic in a consistent, extensible way.  

### Key Features

*   ✅ **Full-Stack Development:** Develop both front-end and back-end components within a single, unified framework.
*   ✅ **Low-Code Approach:** Reduce development time with built-in features and automations.
*   ✅ **Built-in Admin Interface:** Easily manage and customize application data with a pre-built admin dashboard.
*   ✅ **Role-Based Permissions:** Implement granular access control and user management to secure your applications.
*   ✅ **REST API:** Generate automatic RESTful APIs for easy integration with other systems and services.
*   ✅ **Customizable Forms and Views:** Create highly customized user interfaces with server-side scripting and client-side JavaScript.
*   ✅ **Report Builder:** Build custom reports with a user-friendly interface, eliminating the need for manual coding.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting (Frappe Cloud)

For a hassle-free deployment experience, consider [Frappe Cloud](https://frappecloud.com). This open-source platform simplifies hosting, upgrades, and maintenance of your Frappe applications, offering a complete developer platform for managing your deployments.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Self-Hosting

#### Docker

**Prerequisites:** docker, docker-compose, git.  Refer to [Docker Documentation](https://docs.docker.com) for Docker setup details.

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

After a few minutes, your site will be accessible on your localhost at port 8080. Use the following default credentials to log in:

*   **Username:** Administrator
*   **Password:** admin

For ARM-based Docker setups, see the [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) instructions.

## Development Setup

### Manual Install

The easiest method is to utilize our install script for bench, which will install all necessary dependencies (e.g., MariaDB). Refer to [bench documentation](https://github.com/frappe/bench) for further information.

The script generates new passwords for the Frappe "Administrator" user, the MariaDB root user, and the frappe user, displaying them and saving them to `~/frappe_passwords.txt`.

### Local Development

Follow these steps to set up the repository locally:

1.  Set up bench using the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:
    ```bash
    bench start
    ```

2.  In a separate terminal window, run:
    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open the URL `http://frappe.localhost:8000/app` in your browser to view the running application.

## Learning and Community

*   📚 [Frappe School](https://frappe.school):  Learn Frappe Framework and ERPNext through community and maintainer-led courses.
*   📖 [Official Documentation](https://docs.frappe.io/framework):  Comprehensive documentation for Frappe Framework.
*   💬 [Discussion Forum](https://discuss.frappe.io/):  Engage with the Frappe Framework community.
*   💻 [buildwithhussain.com](https://buildwithhussain.com):  Watch Frappe Framework in action to build web apps.

## Contributing

*   📜 [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   🛡️ [Report Security Vulnerabilities](https://frappe.io/security)
*   🤝 [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   🌐 [Translations](https://crowdin.com/project/frappe)

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