<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Your Business</h2>
    <p align="center">
        <p>Empower your business with ERPNext, a powerful, intuitive, and open-source Enterprise Resource Planning (ERP) system.</p>
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

## ERPNext: Your All-in-One Business Solution

**ERPNext** is a 100% open-source ERP system designed to streamline your business operations, offering a comprehensive suite of tools to manage various aspects of your company. This includes everything from accounting and inventory management to manufacturing and project tracking.

### Key Features:

*   **Accounting:** Simplify your financial management with tools for transactions, reporting, and analysis.
*   **Order Management:** Efficiently manage sales orders, customers, suppliers, inventory, and fulfillment.
*   **Manufacturing:** Optimize your production cycle with features for material consumption, capacity planning, and subcontracting.
*   **Asset Management:** Track and manage your organization's assets from purchase to disposal, encompassing IT infrastructure and equipment.
*   **Projects:** Deliver internal and external projects on time and within budget with project management tools, including task and issue tracking.

<details open>
<summary>More Features</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Under the Hood

ERPNext is built upon the following key components:

*   **Frappe Framework:** A full-stack web application framework, providing a robust foundation for web applications, including a database abstraction layer, user authentication, and a REST API. ([Frappe Framework GitHub](https://github.com/frappe/frappe))
*   **Frappe UI:** A Vue.js-based UI library that provides a modern and user-friendly interface. ([Frappe UI GitHub](https://github.com/frappe/frappe-ui))

---

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for a hassle-free hosting experience. It provides easy installation, upgrades, monitoring, and support for your Frappe applications.

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

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the Docker Compose command:
    ```bash
    docker compose -f pwd.yml up -d
    ```

Your site should be accessible on localhost:8080 after a few minutes. Use the default login credentials: Username: `Administrator`, Password: `admin`.

Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

---

## Development Setup

### Manual Install

**The Easy Way:** Use the install script for bench, which handles all dependencies (e.g., MariaDB). See [bench documentation](https://github.com/frappe/bench).
New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

To set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
    ```bash
    bench start
    ```
2.  In a separate terminal:
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
4.  Open `http://erpnext.localhost:8000/app` in your browser.

---

## Learning and Community

1.  [Frappe School](https://school.frappe.io) - Learn ERPNext and Frappe Framework.
2.  [Official documentation](https://docs.erpnext.com/)
3.  [Discussion Forum](https://discuss.erpnext.com/)
4.  [Telegram Group](https://erpnext_public.t.me)

---

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

---

## Logo and Trademark Policy

Read our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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