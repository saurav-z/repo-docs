<!-- ERPNext: The Open-Source ERP for Modern Businesses -->
<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP Software</h2>
    <p align="center">
        <p><b>Transform your business with ERPNext, the powerful, intuitive, and 100% open-source ERP system.</b></p>
    </p>

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)<br><br>
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)
<br>
<a href="https://github.com/frappe/erpnext">View the Source Code on GitHub</a>

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

ERPNext is a comprehensive, 100% open-source Enterprise Resource Planning (ERP) system designed to empower businesses of all sizes. Manage your entire business operations—from accounting to manufacturing—with a single, integrated platform.

## Key Features

*   **Accounting:** Streamline your finances with tools for transaction recording, financial reporting, and cash flow management.
*   **Order Management:** Efficiently manage inventory, sales orders, customer relationships, and supply chain logistics.
*   **Manufacturing:** Simplify production cycles, track material usage, manage capacity planning, and handle subcontracting.
*   **Asset Management:** Track assets from purchase to disposal, covering IT infrastructure, equipment, and more.
*   **Project Management:** Deliver projects on time and within budget with project tracking, task management, and timesheet integration.

<details open>
<summary>More Features</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Under the Hood

ERPNext is built upon the robust [Frappe Framework](https://github.com/frappe/frappe), a full-stack web application framework, and a modern user interface via [Frappe UI](https://github.com/frappe/frappe-ui).

## Getting Started

### Production Setup

Choose from the following deployment options:

*   **Managed Hosting (Frappe Cloud):** For ease of use, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform that handles installation, upgrades, monitoring, and support.

<div>
	<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

*   **Self-Hosted:** For self-hosting, choose between Docker and Manual Install.

#### Docker

1.  **Prerequisites:** Docker, Docker Compose, Git.
2.  **Run:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your site on `localhost:8080` with the default credentials:

*   Username: `Administrator`
*   Password: `admin`

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

### Development Setup

#### Manual Install

1.  **Bench Setup:** Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to setup bench and start the server.
2.  **New Site:**
    ```bash
    bench new-site erpnext.localhost
    ```
3.  **Get and Install ERPNext:**
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
4.  Access the app at `http://erpnext.localhost:8000/app`.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Courses to learn Frappe Framework and ERPNext.
*   [Official documentation](https://docs.erpnext.com/) - Comprehensive documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Please review the [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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