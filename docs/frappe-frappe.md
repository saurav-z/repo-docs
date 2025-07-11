<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>

 **Build powerful, real-world web applications quickly with Frappe, a low-code framework built on Python and JavaScript.**
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

## Frappe Framework: The Low-Code Powerhouse

Frappe Framework is a full-stack web application framework that simplifies development with its Python and JavaScript core. It provides a robust, integrated environment for building dynamic and scalable applications.  Learn more and contribute on the [official GitHub repository](https://github.com/frappe/frappe).

### Key Features:

*   **Full-Stack Development:** Develop both front-end and back-end with a single, unified framework.
*   **Built-in Admin Interface:** Reduce development time with a pre-built, customizable admin dashboard.
*   **Role-Based Permissions:** Implement granular access control for enhanced security.
*   **REST API Generation:**  Automatically create RESTful APIs for seamless integration.
*   **Customizable Forms and Views:** Tailor your application's interface with server-side scripting and client-side JavaScript.
*   **Report Builder:** Empower users to generate custom reports without coding.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

For a hassle-free deployment experience, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting your Frappe applications. Frappe Cloud handles installation, upgrades, monitoring, maintenance, and support.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Self Hosting

**Docker**

1.  **Prerequisites:** Ensure you have Docker, docker-compose, and git installed. Refer to [Docker Documentation](https://docs.docker.com) for setup details.

2.  **Run the following commands:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

3.  **Access:** After a few minutes, your site will be accessible on `localhost:8080`.
    *   **Login Credentials:**
        *   Username: `Administrator`
        *   Password: `admin`

4.  **ARM64 Architecture:**  See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for specific instructions for ARM-based Docker setups.

## Development Setup

### Manual Install

The Easy Way: Use the install script for bench to install all dependencies (e.g. MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

1.  **Bench Setup:** Follow the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) to set up bench and start the server:

```bash
bench start
```

2.  **New Site:** In a separate terminal, create a new site:

```bash
bench new-site frappe.localhost
```

3.  **Access Your App:** Open `http://frappe.localhost:8000/app` in your browser.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Courses for Frappe Framework and ERPNext.
2.  [Official documentation](https://docs.frappe.io/framework) - Comprehensive documentation.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - See Frappe in action building web apps.

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