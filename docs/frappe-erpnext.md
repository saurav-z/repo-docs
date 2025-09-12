# ERPNext: Open-Source ERP for Growing Businesses

**Unlock the power of streamlined business operations with ERPNext, a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system.**  [Explore the original repository](https://github.com/frappe/erpnext).

<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
</div>

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

## Key Features

ERPNext provides a comprehensive suite of tools to manage every aspect of your business:

*   **Accounting:** Manage your finances with tools for transactions, financial reports, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales orders, customers, suppliers, shipments, and fulfillment.
*   **Manufacturing:** Simplify the production cycle, track material consumption, and manage capacity planning.
*   **Asset Management:** Manage assets from purchase to disposal, covering all aspects of your organization's infrastructure and equipment.
*   **Projects:**  Deliver projects on time, within budget, and profitably by tracking tasks, timesheets, and issues.

<details open>
<summary>More</summary>
    <img src="https://erpnext.com/files/v16_bom.png"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png"/>
    <img src="https://erpnext.com/files/v16_job_card.png"/>
    <img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Under the Hood

ERPNext is built on robust open-source technologies:

*   **Frappe Framework:** A full-stack web application framework (Python and Javascript) providing the foundation for ERPNext.
*   **Frappe UI:** A Vue-based UI library providing a modern and user-friendly interface.

## Production Setup

Choose the hosting option that best suits your needs:

### Managed Hosting

Simplify your ERPNext deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform that handles installation, upgrades, and maintenance.

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

**Steps:**

1.  Clone the repository: `git clone https://github.com/frappe/frappe_docker`
2.  Navigate to the directory: `cd frappe_docker`
3.  Run the Docker Compose command: `docker compose -f pwd.yml up -d`

Access your site at `localhost:8080`. Use the default credentials: Administrator / admin.

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

For detailed installation instructions, refer to the [Installation Steps](https://frappeframework.com/docs/user/en/installation) on the Frappe Framework documentation.  A quick way to get started is by using the install script for bench, which will install all dependencies.

### Local

1.  Set up bench and start the server:
    ```bash
    bench start
    ```
2.  In a separate terminal, create a new site:
    ```bash
    bench new-site erpnext.localhost
    ```
3.  Get and install the ERPNext app:
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
4.  Access the app in your browser at `http://erpnext.localhost:8000/app`.

## Learning and Community

Stay connected with the ERPNext community for support and knowledge:

1.  [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext.
2.  [Official documentation](https://docs.erpnext.com/) - Extensive documentation.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with other users and providers.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from the community.

## Contributing

Contribute to the ERPNext project:

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Review the [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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