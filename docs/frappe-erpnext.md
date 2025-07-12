<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Businesses of All Sizes</h2>
    <p align="center">
        <b>Empower your business with ERPNext, a powerful, intuitive, and completely free open-source ERP solution.</b>
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

## What is ERPNext?

ERPNext is a 100% open-source Enterprise Resource Planning (ERP) system designed to help you streamline and manage all aspects of your business operations. From accounting and inventory to manufacturing and project management, ERPNext provides a comprehensive suite of tools in a single, integrated platform.

### Key Features

*   **Accounting:** Comprehensive tools for managing cash flow, from transaction recording to financial reporting and analysis.
*   **Order Management:** Efficiently track inventory, manage sales orders, handle customer and supplier relationships, and fulfill orders.
*   **Manufacturing:** Simplify your production cycle with tools for tracking material consumption, capacity planning, and subcontracting.
*   **Asset Management:** Track assets throughout their lifecycle, covering everything from IT infrastructure to equipment.
*   **Projects:** Deliver projects on time, within budget, and profitably. Track tasks, timesheets, and issues by project.

<details open>
<summary>More Features</summary>
    <img src="https://erpnext.com/files/v16_bom.png"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png"/>
    <img src="https://erpnext.com/files/v16_job_card.png"/>
    <img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

---

## Under the Hood

ERPNext is built on robust open-source technologies:

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework written in Python and Javascript, providing the foundation for ERPNext with features like a database abstraction layer, user authentication, and a REST API.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library that delivers a modern and user-friendly interface for the ERPNext application.

---

## Getting Started

### Production Setup

Choose the deployment method that best suits your needs:

*   **Managed Hosting (Recommended):** Simplify your ERPNext setup with [Frappe Cloud](https://frappecloud.com).  It handles installation, upgrades, and maintenance, allowing you to focus on your business.

<div>
	<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

*   **Self-Hosted:**  Deploy ERPNext on your own infrastructure using Docker.

#### Self-Hosted: Docker

**Prerequisites:** docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

**Instructions:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a few minutes, your site should be accessible on `localhost:8080`. Use the following default login credentials to access the site:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

---

### Development Setup

#### Manual Install

The Easy Way: our install script for bench will install all dependencies (e.g. MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

#### Local Development Setup

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, run these commands:

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

4.  Open `http://erpnext.localhost:8000/app` in your browser to run the app.

---

## Learn and Get Involved

*   [**Frappe School**](https://school.frappe.io): Learn the Frappe Framework and ERPNext through courses created by the maintainers and community.
*   [**Official Documentation**](https://docs.erpnext.com/): Comprehensive documentation for ERPNext.
*   [**Discussion Forum**](https://discuss.erpnext.com/): Engage with the ERPNext community and service providers.
*   [**Telegram Group**](https://erpnext_public.t.me): Get instant help from a large community of users.

---

## Contribute

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