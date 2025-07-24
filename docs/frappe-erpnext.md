<!-- ERPNext - The Open-Source ERP for Your Business -->

<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP Software</h2>
    <p align="center">
        <b>Manage your entire business with ERPNext, a powerful and intuitive open-source Enterprise Resource Planning (ERP) system.</b>
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
	-
	<a href="https://github.com/frappe/erpnext">View on GitHub</a>
</div>

---

## About ERPNext

ERPNext is a 100% open-source ERP system designed to streamline your business operations. From accounting to manufacturing, ERPNext provides a comprehensive solution for businesses of all sizes.

### Key Features of ERPNext:

*   **Accounting:** Manage finances with tools for transactions, reporting, and analysis.
*   **Order Management:** Track inventory, manage sales, and fulfill orders efficiently.
*   **Manufacturing:** Simplify production cycles, track materials, and manage capacity planning.
*   **Asset Management:** Oversee assets from purchase to disposal, across all departments.
*   **Projects:** Manage projects, track tasks, and monitor profitability.

<details open>

<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

---

## Technology Under the Hood

*   **Frappe Framework:** A full-stack web application framework (Python and Javascript) providing the foundation for ERPNext.  [Learn more](https://github.com/frappe/frappe)
*   **Frappe UI:** A Vue.js-based UI library for a modern and user-friendly interface. [Learn more](https://github.com/frappe/frappe-ui)

---

## Deployment Options

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), the official hosting platform.  It handles installation, upgrades, monitoring, and maintenance.

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

Prerequisites: Docker, docker-compose, and Git.

To get started:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your ERPNext instance at `http://localhost:8080`.

*   **Login:**
    *   Username: Administrator
    *   Password: admin

Refer to the [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) documentation for ARM-based setups.

---

## Development Setup

### Manual Install

See [bench install guide](https://github.com/frappe/bench) for bench details.

### Local

To set up a local development environment:

1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up bench.
2.  Start the server:
    ```bash
    bench start
    ```
3.  In a separate terminal, run:
    ```bash
    bench new-site erpnext.localhost
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
4.  Access your local ERPNext instance at `http://erpnext.localhost:8000/app`.

---

## Resources for Learning and Community

1.  [Frappe School](https://school.frappe.io) - Courses on Frappe Framework and ERPNext.
2.  [Official documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from other users.

---

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

---

## Logo and Trademark Policy

Please review our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

---

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