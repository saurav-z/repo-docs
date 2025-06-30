<div align="center">
    <a href="https://frappe.io/erpnext">
	    <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
</div>

# ERPNext: Open-Source ERP for Growing Businesses

**ERPNext is a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system that empowers businesses to streamline operations and drive growth.** [Explore the original repository](https://github.com/frappe/erpnext).

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

ERPNext provides a comprehensive suite of modules to manage your business operations efficiently:

*   **Accounting:** Manage cash flow, track transactions, and generate financial reports.
*   **Order Management:** Track inventory, manage sales orders, and handle order fulfillment.
*   **Manufacturing:** Simplify the production cycle, manage bills of materials, and track material consumption.
*   **Asset Management:** Track assets from purchase to disposal, covering IT infrastructure and equipment.
*   **Projects:** Manage both internal and external projects, track tasks, timesheets, and profitability.

<details open>
    <summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>


## Technology Stack

ERPNext is built on robust open-source technologies:

*   **Frappe Framework:** A full-stack web application framework (Python & JavaScript) providing the foundation.
*   **Frappe UI:** A Vue.js-based UI library for a modern user interface.

## Deployment Options

### Managed Hosting

Frappe Cloud offers a user-friendly platform for hosting your ERPNext applications.  It simplifies installation, upgrades, and maintenance.

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

**Prerequisites:** Docker, docker-compose, and git.

**Steps:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a couple of minutes, your site should be accessible on `localhost:8080`.  Use the default login credentials:

*   **Username:** Administrator
*   **Password:** admin

See the [Frappe Docker documentation](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

1.  Follow [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up bench and start the server:
    ```bash
    bench start
    ```

2.  In a separate terminal:
    ```bash
    bench new-site erpnext.localhost
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```

3.  Open `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community Resources

*   [Frappe School](https://school.frappe.io):  Learn Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.erpnext.com/): Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me): Get instant help from the community.

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