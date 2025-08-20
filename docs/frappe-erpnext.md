# ERPNext: Open-Source ERP for Businesses of All Sizes

**Tired of juggling multiple software solutions? ERPNext is a powerful, open-source ERP system that streamlines your business operations.**  [Explore the original repository](https://github.com/frappe/erpnext).

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

![ERPNext Hero Image](./erpnext/public/images/v16/hero_image.png)

[Live Demo](https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo) | [Website](https://frappe.io/erpnext) | [Documentation](https://docs.frappe.io/erpnext/)

## Key Features of ERPNext

ERPNext is a comprehensive ERP solution, offering a wide array of features to manage your business effectively:

*   **Accounting:**  Manage your finances with ease, from transaction recording to financial reporting and analysis.
*   **Order Management:** Track inventory, manage sales and purchase orders, and optimize order fulfillment.
*   **Manufacturing:** Simplify your production cycle, track material consumption, and manage subcontracting.
*   **Asset Management:**  Track and manage your assets, from IT infrastructure to equipment, throughout their lifecycle.
*   **Projects:** Manage both internal and external projects efficiently, tracking tasks, timesheets, and profitability.

<details open>
<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Under the Hood

ERPNext is built upon robust open-source technologies:

*   **Frappe Framework:**  A full-stack web application framework written in Python and Javascript, providing the foundation for ERPNext. ([Frappe Framework](https://github.com/frappe/frappe))
*   **Frappe UI:** A Vue-based UI library provides a modern and user-friendly interface. ([Frappe UI](https://github.com/frappe/frappe-ui))

## Production Setup

Choose the best way to run ERPNext for you:

### Managed Hosting
Simplify deployment with Frappe Cloud, a user-friendly platform that takes care of installation, upgrades, monitoring, and maintenance.

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

Quickly get up and running using Docker:

**Prerequisites:** Docker, Docker Compose, and Git.

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your ERPNext instance at `http://localhost:8080` using the default credentials:

*   **Username:** Administrator
*   **Password:** admin

Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setups.

## Development Setup

### Manual Install

Refer to the [Frappe Framework Installation Steps](https://frappeframework.com/docs/user/en/installation) for detailed instructions on setting up a local development environment.  You'll need to use `bench start` in a separate terminal. Then, follow these steps:

```bash
# Create a new site
bench new-site erpnext.localhost

# Get the ERPNext app
bench get-app https://github.com/frappe/erpnext

# Install the app
bench --site erpnext.localhost install-app erpnext
```

Access your development instance at `http://erpnext.localhost:8000/app`.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Comprehensive courses on Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.erpnext.com/) - In-depth documentation for ERPNext.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community and find solutions.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from a large user community.

## Contributing

We welcome contributions! Please review the following guidelines:

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