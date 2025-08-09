<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
</div>

# ERPNext: Open-Source ERP Software for Your Business

**ERPNext is a powerful, intuitive, and open-source ERP system that empowers businesses of all sizes to streamline operations and drive growth.**  [View the original repository](https://github.com/frappe/erpnext).

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)<br><br>
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
</div>

## Key Features of ERPNext

ERPNext offers a comprehensive suite of features to manage your business:

*   **Accounting:** Manage your finances from transactions to financial reports.
*   **Order Management:** Track inventory, manage sales, and fulfill orders efficiently.
*   **Manufacturing:** Simplify production cycles and track material consumption.
*   **Asset Management:** Track assets from purchase to retirement.
*   **Projects:** Manage projects on time, on budget and profitably by tracking tasks, timesheets, and issues.

<details open>
<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Technology Behind ERPNext

ERPNext is built on the following technologies:

*   **Frappe Framework:** A full-stack web application framework (Python/JavaScript) providing the foundation for building web applications.
*   **Frappe UI:** A Vue-based UI library, providing a modern user interface.

## Production Setup

### Managed Hosting

Simplify your ERPNext deployment with [Frappe Cloud](https://frappecloud.com).  It handles installation, upgrades, and maintenance.

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

**Prerequisites:** docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

**Instructions:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your site on `localhost:8080` using the default credentials:

*   Username: Administrator
*   Password: admin

For ARM-based Docker setup, see [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Install

The Easy Way: our install script for bench will install all dependencies (e.g. MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local Setup

Follow these steps to set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server
    ```bash
    bench start
    ```

2.  In a separate terminal window, run the following commands:
    ```bash
    # Create a new site
    bench new-site erpnext.localhost
    ```

3.  Get the ERPNext app and install it
    ```bash
    # Get the ERPNext app
    bench get-app https://github.com/frappe/erpnext

    # Install the app
    bench --site erpnext.localhost install-app erpnext
    ```

4.  Open the URL `http://erpnext.localhost:8000/app` in your browser to access the running app.

## Learning and Community Resources

*   [Frappe School](https://school.frappe.io) - Learn ERPNext and Frappe Framework.
*   [Official documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from a large user community.

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