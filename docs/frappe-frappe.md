<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>
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

## Frappe Framework: Build Powerful Web Applications, Fast

Frappe Framework is a full-stack, low-code web application framework built for real-world applications using Python and JavaScript.  This framework provides developers with the tools needed to build robust and scalable web applications quickly.

**[View the Frappe Framework Repository on GitHub](https://github.com/frappe/frappe)**

### Key Features

*   **Full-Stack Development:** Develop both front-end and back-end components within a single framework, streamlining the development process.
*   **Built-in Admin Interface:** Save time and effort with a pre-built, customizable admin dashboard for efficient data management.
*   **Role-Based Permissions:** Implement granular access control with a comprehensive user and role management system.
*   **REST API Generation:** Automatically generate RESTful APIs for all models, enabling seamless integration with other services.
*   **Customizable Forms and Views:** Tailor forms and views to your specific needs using server-side scripting and client-side JavaScript.
*   **Powerful Report Builder:** Empower users to create custom reports without requiring any coding knowledge.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Simplify your deployments with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. Frappe Cloud handles installation, upgrades, monitoring, and support, providing a comprehensive developer platform for managing multiple Frappe deployments.

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

To self-host using Docker, ensure you have Docker, Docker Compose, and Git installed.

Run the following commands:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your site at `localhost:8080` after a few minutes, using the default credentials:

*   **Username:** Administrator
*   **Password:** admin

Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The easiest way to install Frappe is using the bench install script, which handles all dependencies, including MariaDB. For details, see the [bench documentation](https://github.com/frappe/bench).

The script will create new passwords for the Frappe "Administrator" user, MariaDB root, and the frappe user (passwords will be displayed and saved in `~/frappe_passwords.txt`).

### Local

To set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:
    ```bash
    bench start
    ```
2.  In a separate terminal window, run:
    ```bash
    bench new-site frappe.localhost
    ```
3.  Open `http://frappe.localhost:8000/app` in your browser.

## Learning and Community

*   **Frappe School:** Learn Frappe Framework and ERPNext through courses by maintainers and the community: [https://frappe.school](https://frappe.school)
*   **Official Documentation:** Explore comprehensive documentation for the Frappe Framework: [https://docs.frappe.io/framework](https://docs.frappe.io/framework)
*   **Discussion Forum:** Engage with the Frappe Framework community: [https://discuss.frappe.io/](https://discuss.frappe.io/)
*   **buildwithhussain.com:** Watch Frappe Framework being used in real-world applications: [https://buildwithhussain.com](https://buildwithhussain.com)

## Contributing

*   **Issue Guidelines:** [https://github.com/frappe/erpnext/wiki/Issue-Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   **Report Security Vulnerabilities:** [https://frappe.io/security](https://frappe.io/security)
*   **Pull Request Requirements:** [https://github.com/frappe/erpnext/wiki/Contribution-Guidelines](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   **Translations:** [https://crowdin.com/project/frappe](https://crowdin.com/project/frappe)

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