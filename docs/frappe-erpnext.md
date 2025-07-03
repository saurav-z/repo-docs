<!-- Improved & SEO-Optimized README -->

<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP Software for Business Management</h2>
    <p align="center">
        <b>Manage your entire business with ERPNext, a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system.</b>
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
    <a href="https://github.com/frappe/erpnext"><b>View on GitHub</b></a>
</div>

---

## What is ERPNext?

ERPNext is a comprehensive, open-source ERP system designed to help businesses streamline operations, improve efficiency, and boost productivity.  It offers a robust set of modules to manage various aspects of your business, all in one centralized platform.

### Key Features of ERPNext

*   **Accounting:** Manage your finances seamlessly with tools for transactions, financial reporting, and cash flow analysis.
*   **Order Management:** Track inventory, handle sales orders, manage customers and suppliers, and ensure efficient order fulfillment.
*   **Manufacturing:** Simplify the production cycle, track material consumption, and optimize capacity planning.
*   **Asset Management:**  Monitor your assets from purchase to disposal, covering IT infrastructure, equipment, and more.
*   **Projects:**  Deliver projects on time, within budget, and profitably, with features for task management, timesheets, and issue tracking.

<details open>
  <summary>More ERPNext Features</summary>
    <img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Under the Hood

ERPNext is built on two core components:

*   **Frappe Framework:** A full-stack web application framework built with Python and JavaScript, providing a strong foundation for web application development. ([Frappe Framework GitHub](https://github.com/frappe/frappe))
*   **Frappe UI:** A Vue-based UI library that delivers a modern and user-friendly interface. ([Frappe UI GitHub](https://github.com/frappe/frappe-ui))

---

## Getting Started with ERPNext

### Managed Hosting (Recommended)

Simplify your ERPNext setup with [Frappe Cloud](https://frappecloud.com).  Frappe Cloud handles installation, upgrades, maintenance, and support, allowing you to focus on your business.

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

**Prerequisites:** Docker, Docker Compose, Git.

To run ERPNext using Docker, follow these steps:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Access your ERPNext instance on `http://localhost:8080`.  Use the following default credentials:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

#### Manual Install

The Easy Way: Use the install script for bench, which will automatically install all dependencies (e.g., MariaDB). See [Bench Documentation](https://github.com/frappe/bench) for more details.

### Local Development

To set up a local development environment:

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server.

    ```bash
    bench start
    ```

2.  In a separate terminal, run these commands:

    ```bash
    # Create a new site
    bench new-site erpnext.localhost
    ```

3.  Get and install the ERPNext app:

    ```bash
    # Get the ERPNext app
    bench get-app https://github.com/frappe/erpnext

    # Install the app
    bench --site erpnext.localhost install-app erpnext
    ```

4.  Open `http://erpnext.localhost:8000/app` in your browser to access your local ERPNext instance.

---

## Learning and Community Resources

*   [Frappe School](https://school.frappe.io): Learn ERPNext and the Frappe Framework.
*   [Official Documentation](https://docs.erpnext.com/):  Comprehensive documentation for ERPNext.
*   [Discussion Forum](https://discuss.erpnext.com/):  Connect with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me):  Get instant help from the user community.

---

## Contributing

We welcome contributions to ERPNext!  Please review these resources:

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

---

## Logo and Trademark Policy

Review our [Logo and Trademark Policy](TRADEMARK_POLICY.md) for usage guidelines.

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