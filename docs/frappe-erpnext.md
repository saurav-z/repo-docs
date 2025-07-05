<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP Software for Growing Businesses</h2>
    <p align="center">
        <p>Empower your business with ERPNext, a powerful, intuitive, and open-source ERP solution that simplifies complex business processes.</p>
    </p>

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)<br><br>
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

</div>

<div align="center">
	<img src="./erpnext/public/images/v16/hero_image.png"/>
</div>

<div align="center">
	<a href="https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo">Live Demo</a>
	-
	<a href="https://frappe.io/erpnext">Website</a>
	-
	<a href="https://docs.frappe.io/erpnext/">Documentation</a>
</div>

## About ERPNext

**ERPNext** is a 100% open-source ERP (Enterprise Resource Planning) system designed to help businesses of all sizes streamline operations and achieve growth.  Manage your entire business, from accounting and inventory to manufacturing and project management, all in one integrated platform.  [Learn more about ERPNext on GitHub](https://github.com/frappe/erpnext).

### Key Features:

*   **Accounting:** Comprehensive tools for managing finances, including transactions, financial reports, and cash flow analysis.
*   **Order Management:** Efficiently track inventory, manage sales orders, handle customer and supplier relationships, and fulfill orders seamlessly.
*   **Manufacturing:** Simplify the production cycle with features like material consumption tracking, capacity planning, and subcontracting management.
*   **Asset Management:** Track assets from purchase to disposal, covering IT infrastructure and equipment across your organization.
*   **Projects:** Deliver internal and external projects on time and within budget by tracking tasks, timesheets, and issues.

<details open>

<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Under the Hood

*   [**Frappe Framework**](https://github.com/frappe/frappe): A powerful full-stack web application framework, providing the foundation for ERPNext.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library that offers a modern and intuitive user interface.

## Getting Started

Choose your preferred setup method:

### Production Setup

#### Managed Hosting

Experience the simplicity of Frappe Cloud, a user-friendly and sophisticated platform for hosting Frappe applications. It takes care of installation, setup, upgrades, monitoring, maintenance, and support of your Frappe deployments.

<div>
	<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

#### Self-Hosted

##### Docker

**Prerequisites:** Docker, Docker Compose, Git. Refer to the [Docker Documentation](https://docs.docker.com) for installation instructions.

**Steps:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a few minutes, your site should be accessible on your localhost at port 8080.  Use the following credentials to log in:

*   **Username:** Administrator
*   **Password:** admin

For ARM-based Docker setups, see [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Install

**The Easy Way:** Use our install script for bench. This will automatically install all dependencies, including MariaDB.  See the [Frappe Bench documentation](https://github.com/frappe/bench) for details.

The script creates new passwords for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user.  These passwords are displayed and saved to `~/frappe_passwords.txt`.

### Local

To set up the repository locally, follow these steps:

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, run the following commands:

    ```bash
    # Create a new site
    bench new-site erpnext.localhost
    ```

3.  Get the ERPNext app and install it:

    ```bash
    # Get the ERPNext app
    bench get-app https://github.com/frappe/erpnext

    # Install the app
    bench --site erpnext.localhost install-app erpnext
    ```

4.  Open `http://erpnext.localhost:8000/app` in your browser. You should see the app running.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Courses on Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.erpnext.com/) - Comprehensive documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from other users.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Please review our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

<br />
<br />
<div align="center" style="padding-top: 0.75rem;">
	<a href="https://frappe.io" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png">
			<img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies" height="28"/>
		</picture>
	</a>
</div>