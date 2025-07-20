<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
    <h2>ERPNext: Open-Source ERP Software</h2>
    <p align="center">
        <b>Simplify your business operations with ERPNext, a powerful and intuitive open-source Enterprise Resource Planning (ERP) system.</b>
    </p>
</div>

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

<div align="center">
	<img src="./erpnext/public/images/v16/hero_image.png"/>
</div>

<div align="center">
	<a href="https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo">Live Demo</a>
	-
	<a href="https://frappe.io/erpnext">Website</a>
	-
	<a href="https://docs.frappe.io/erpnext/">Documentation</a>
	-
	<a href="https://github.com/frappe/erpnext"><b>View on GitHub</b></a>
</div>

## About ERPNext

ERPNext is a 100% open-source ERP system designed to help businesses manage all core functions in a single platform. From accounting to manufacturing, ERPNext provides a comprehensive suite of tools to streamline operations and boost efficiency.

### Key Features

*   **Accounting:** Manage your finances with comprehensive tools, including transaction recording, financial reporting, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales orders, and fulfill customer orders efficiently.
*   **Manufacturing:** Simplify your production cycle with tools for tracking material consumption, capacity planning, and subcontracting.
*   **Asset Management:** Track and manage all your company assets, from IT infrastructure to equipment.
*   **Projects:** Manage both internal and external projects, ensuring timely delivery, budget adherence, and profitability.

<details open>
<summary>More</summary>
    <img src="https://erpnext.com/files/v16_bom.png"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png"/>
    <img src="https://erpnext.com/files/v16_job_card.png"/>
    <img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Under the Hood

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework written in Python and Javascript, providing a robust foundation for building web applications.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library that provides a modern user interface.

## Deployment Options

### Managed Hosting

[Frappe Cloud](https://frappecloud.com) offers a simple and user-friendly platform for hosting Frappe applications. It takes care of installation, setup, upgrades, monitoring, maintenance, and support, allowing you to focus on your business.

<div>
	<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

### Self-Hosted

#### Docker

Prerequisites: Docker, Docker Compose, Git. Refer to [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

To get started with Docker:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Your site should be accessible on your localhost at port 8080 after a few minutes. Use the following default credentials:

*   Username: Administrator
*   Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

Our install script for bench will install all dependencies (e.g., MariaDB). See [bench documentation](https://github.com/frappe/bench) for details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local Development

Follow these steps to set up the repository locally:

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

4.  Open the URL `http://erpnext.localhost:8000/app` in your browser. You should see the app running.

## Learning and Community

1.  [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext through courses.
2.  [Official documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from the community.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

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