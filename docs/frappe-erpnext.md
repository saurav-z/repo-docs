<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
</div>

# ERPNext: Open-Source ERP for Growing Businesses

ERPNext is a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system that helps businesses streamline operations and boost efficiency.  **[Explore ERPNext on GitHub](https://github.com/frappe/erpnext)**.

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


## Key Features of ERPNext:

*   **Accounting:** Manage your finances with tools for transaction recording, financial reporting, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales orders, handle customer and supplier relationships, and optimize fulfillment.
*   **Manufacturing:** Simplify your production cycle with features for production planning, material tracking, and subcontracting.
*   **Asset Management:** Track assets from purchase to disposal, covering IT infrastructure and equipment across your organization.
*   **Projects:** Manage projects on time and on budget. Track tasks, timesheets, and issues.

<details open>

<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Core Technologies

*   **Frappe Framework:** A full-stack web application framework (Python and Javascript) providing a robust foundation, database abstraction, user authentication, and REST API.
*   **Frappe UI:** A Vue-based UI library for a modern, user-friendly interface.

## Deployment Options

### Managed Hosting

Simplify deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. It handles installation, upgrades, monitoring, and maintenance.

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

**Prerequisites:** Docker, Docker Compose, and Git.

**Steps:**

1.  Clone the repository: `git clone https://github.com/frappe/frappe_docker`
2.  Navigate to the directory: `cd frappe_docker`
3.  Run Docker Compose: `docker compose -f pwd.yml up -d`

Access your site on `localhost:8080` using the default credentials:
*   **Username:** Administrator
*   **Password:** admin

*For ARM-based Docker setups, see the [Frappe Docker documentation](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).*

## Development Setup

### Manual Install

*   Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server: `bench start`
*   In a separate terminal:

    *   Create a new site: `bench new-site erpnext.localhost`
    *   Get the ERPNext app: `bench get-app https://github.com/frappe/erpnext`
    *   Install the app: `bench --site erpnext.localhost install-app erpnext`
*   Open `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community Resources

1.  [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext.
2.  [Official documentation](https://docs.erpnext.com/) - Extensive ERPNext documentation.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Connect with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help.

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