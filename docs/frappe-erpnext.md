<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Business Growth</h2>
    <p align="center">
        <p>Empower your business with ERPNext, a powerful, intuitive, and 100% open-source ERP system.</p>
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

## ERPNext: The Open-Source ERP Solution

**ERPNext** is a comprehensive, open-source Enterprise Resource Planning (ERP) system designed to help businesses of all sizes manage their operations efficiently.  This platform provides a complete suite of tools, from accounting to manufacturing, all in one integrated system.  Visit the [ERPNext GitHub repository](https://github.com/frappe/erpnext) for more information.

### Key Features

*   **Accounting:** Manage your finances with comprehensive tools for transactions, reporting, and analysis.
*   **Order Management:** Track inventory, manage sales orders, and streamline fulfillment processes.
*   **Manufacturing:** Simplify production cycles, manage material consumption, and optimize capacity planning.
*   **Asset Management:** Track assets from purchase to disposal, covering all aspects of your organization's infrastructure.
*   **Projects:** Manage both internal and external projects with tools for tracking tasks, timesheets, and profitability.

<details open>
<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Technology Stack

*   **Frappe Framework:** A full-stack web application framework (Python/JavaScript) providing the foundation for ERPNext. ([Frappe Framework GitHub](https://github.com/frappe/frappe))
*   **Frappe UI:** A Vue.js based UI library for a modern and user-friendly interface. ([Frappe UI GitHub](https://github.com/frappe/frappe-ui))

## Installation and Setup

### Managed Hosting

Experience the simplicity of [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.  It handles installation, upgrades, monitoring, and more.

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

**Prerequisites:** Docker, docker-compose, git.  See the [Docker Documentation](https://docs.docker.com) for setup.

**Installation:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your site on `localhost:8080` using the default credentials:

*   **Username:** Administrator
*   **Password:** admin

**Note:** For ARM-based Docker setups, see the [Frappe Docker documentation](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

#### Manual Install

Install dependencies (e.g., MariaDB) using the provided install script.

### Local Development

1.  **Bench Setup:** Follow the [Frappe Framework Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server with `bench start`.
2.  **New Site:** Open a separate terminal and create a new site:
    ```bash
    bench new-site erpnext.localhost
    ```
3.  **Get and Install App:**
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
4.  **Access:** Open `http://erpnext.localhost:8000/app` in your browser to see the running app.

## Learn and Connect

*   [Frappe School](https://school.frappe.io): Learn ERPNext and the Frappe Framework.
*   [Official Documentation](https://docs.erpnext.com/): Extensive documentation.
*   [Discussion Forum](https://discuss.erpnext.com/): Engage with the community.
*   [Telegram Group](https://erpnext_public.t.me): Get instant support.

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