<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
</div>

# ERPNext: Open-Source ERP Software for Growing Businesses

**ERPNext is a powerful and intuitive open-source Enterprise Resource Planning (ERP) system designed to streamline your business operations.**

[View the original repository on GitHub](https://github.com/frappe/erpnext)

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

ERPNext provides a comprehensive suite of features to manage various aspects of your business:

*   **Accounting:** Manage your finances with tools for transaction recording, financial reporting, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales orders, and oversee customer and supplier interactions.
*   **Manufacturing:** Streamline your production cycle, including material consumption tracking and capacity planning.
*   **Asset Management:** Track assets from purchase to disposal.
*   **Projects:** Manage projects, track tasks, timesheets, and profitability.

<details open>
<summary>More Features</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Technology Stack

ERPNext is built upon robust open-source technologies:

*   **Frappe Framework:** A full-stack web application framework (Python and Javascript) providing the foundation for ERPNext, including database abstraction, user authentication, and a REST API. ([Frappe Framework GitHub](https://github.com/frappe/frappe))
*   **Frappe UI:** A Vue.js-based UI library for a modern and user-friendly interface. ([Frappe UI GitHub](https://github.com/frappe/frappe-ui))

## Production Setup

Choose the best setup method for your needs:

### Managed Hosting

**Frappe Cloud** offers a hassle-free managed hosting solution for Frappe applications, handling installation, updates, and maintenance.

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

1.  Clone the repository:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
```

2.  Run using Docker Compose:

```bash
docker compose -f pwd.yml up -d
```

Access your site on `localhost:8080` with the following credentials:

*   Username: `Administrator`
*   Password: `admin`

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

Install using the bench installer, which sets up dependencies:  See https://github.com/frappe/bench for more details.

Passwords will be generated for the ERPNext "Administrator" user, MariaDB root user, and frappe user.

### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:

```bash
bench start
```

2.  In a separate terminal:

```bash
# Create a new site
bench new-site erpnext.localhost
# Get the ERPNext app
bench get-app https://github.com/frappe/erpnext
# Install the app
bench --site erpnext.localhost install-app erpnext
```

3.  Open `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn from courses on Frappe Framework and ERPNext.
*   [Official documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from the user community.

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