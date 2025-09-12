<div align="center">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework: Low-Code Web Development</h1>
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

## Frappe Framework: Build Real-World Web Apps with Ease

**Frappe Framework** is a powerful, low-code web framework built with Python and JavaScript, enabling developers to rapidly build and deploy robust applications. 

🔗 **[Explore the Frappe Framework on GitHub](https://github.com/frappe/frappe)**

### Key Features of Frappe Framework

*   **Full-Stack Development:** Develop both front-end and back-end components within a single framework using Python and JavaScript.
*   **Simplified Administration:** Built-in admin interface for easy data management and application control.
*   **Role-Based Access Control (RBAC):** Comprehensive user and role management system to define and manage permissions and access.
*   **RESTful API Generation:** Automatic REST API generation for easy integration with other systems.
*   **Customizable UI:** Create and customize forms and views using server-side scripting and client-side JavaScript.
*   **Powerful Reporting:** Intuitive report builder for generating custom reports without code.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started

### Production Setup

#### Frappe Cloud
For a hassle-free Frappe application deployment, consider [Frappe Cloud](https://frappecloud.com).

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

#### Self Hosting

### Docker

**Prerequisites:** docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run the Docker Compose file:

    ```bash
    docker compose -f pwd.yml up -d
    ```

    After a few minutes, your site should be accessible on your localhost at port 8080.  Use the default login credentials:

    *   Username: `Administrator`
    *   Password: `admin`

    See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setups.

## Development Setup

### Manual Install

1.  **Bench Setup**: Follow the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) to install bench and start the server.
    ```bash
    bench start
    ```

2.  **Create a New Site**: In a separate terminal window, run:

    ```bash
    bench new-site frappe.localhost
    ```

3.  **Access the App**: Open `http://frappe.localhost:8000/app` in your browser.

## Learn and Contribute

### Resources
1.  [Frappe School](https://frappe.school) - Courses on Frappe Framework and ERPNext.
2.  [Official Documentation](https://docs.frappe.io/framework) - Comprehensive documentation.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - See Frappe Framework in action.

### Contribute
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