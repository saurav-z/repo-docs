<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework: Build Powerful Web Applications Faster</h1>
    <p><b>Frappe Framework</b> empowers developers to create real-world web applications efficiently using Python and JavaScript.</p>
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="License: MIT"></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Code Coverage"></a>
</div>
<div align="center">
	<img src=".github/hero-image.png" alt="Frappe Framework Hero Image" />
</div>
<div align="center">
    <a href="https://frappe.io/framework">Website</a> |
    <a href="https://docs.frappe.io/framework">Documentation</a> |
    <a href="https://github.com/frappe/frappe">View on GitHub</a>
</div>

## Frappe Framework: The Low-Code Web Framework

Frappe Framework is a full-stack web application framework built to accelerate the development of complex, data-driven applications. It leverages Python and MariaDB on the server-side and a tightly integrated client-side library, providing a powerful and efficient development experience. Inspired by the Semantic Web, Frappe emphasizes metadata-driven application design, resulting in highly consistent and extensible applications.

### Key Features

*   **Full-Stack Development:** Develop both the front-end and back-end of your applications with a single framework using Python and JavaScript.
*   **Built-in Admin Interface:** Save time and effort with a customizable admin dashboard for managing application data, user roles, and permissions.
*   **Role-Based Permissions:** Robust user and role management for granular control over application access.
*   **REST API:**  Automated RESTful API generation for seamless integration with other services and systems.
*   **Customizable Forms and Views:** Flexibility in designing forms and views with server-side scripting and client-side JavaScript.
*   **Report Builder:** Easily create custom reports without writing code, empowering users with powerful data visualization tools.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started: Production Setup

### Managed Hosting: Frappe Cloud

For a hassle-free experience, consider [Frappe Cloud](https://frappecloud.com).  It is a user-friendly, open-source platform for hosting Frappe applications. Frappe Cloud handles installation, upgrades, monitoring, and maintenance, providing a fully-featured developer platform.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Self Hosting: Docker

**Prerequisites:** Docker, Docker Compose, Git.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  **Start the containers:**
    ```bash
    docker compose -f pwd.yml up -d
    ```

Access your site on `localhost:8080` using the default credentials:

*   **Username:** Administrator
*   **Password:** admin

For ARM-based Docker setups, refer to the [Frappe Docker documentation](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Installation

The recommended method is using the provided bench install script. For detailed instructions, see [Bench Installation](https://docs.frappe.io/framework/user/en/installation).

**Local Development Steps:**

1.  **Install Bench:** Following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation).
2.  **Start the server:**
    ```bash
    bench start
    ```
3.  **Create a new site:**
    ```bash
    bench new-site frappe.localhost
    ```
4.  **Access the application:** Open `http://frappe.localhost:8000/app` in your browser.

## Learning and Community Resources

*   [Frappe School](https://frappe.school): Courses and tutorials for learning Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.frappe.io/framework): Comprehensive documentation for Frappe Framework.
*   [Discussion Forum](https://discuss.frappe.io/): Engage with the Frappe Framework community.
*   [buildwithhussain.com](https://buildwithhussain.com):  Examples of Frappe Framework in use for building web applications.

## Contributing

We welcome contributions!

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