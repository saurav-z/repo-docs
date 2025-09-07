<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Growing Businesses</h2>
    <p align="center">
        <p>Empowering businesses with a powerful, intuitive, and completely open-source ERP solution.</p>
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
    - <a href="https://github.com/frappe/erpnext">View on GitHub</a>
</div>

## About ERPNext

ERPNext is a 100% open-source Enterprise Resource Planning (ERP) system, designed to streamline your business operations and help you succeed. Manage all aspects of your business, from accounting and inventory to manufacturing and project management, all in one integrated platform.

## Key Features

*   **Accounting:** Comprehensive tools for managing your finances, including transactions, financial reports, and cash flow management.
*   **Order Management:** Track inventory, manage sales orders, handle customer and supplier interactions, and oversee order fulfillment.
*   **Manufacturing:** Simplify your production cycle with tools for material consumption tracking, capacity planning, and subcontracting management.
*   **Asset Management:** Track your organization's assets, from IT infrastructure to equipment, throughout their lifecycle.
*   **Projects:** Deliver projects on time, within budget, and profitably by tracking tasks, timesheets, and issues.

<details open>
<summary>More Screenshots</summary>
    <img src="https://erpnext.com/files/v16_bom.png"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png"/>
    <img src="https://erpnext.com/files/v16_job_card.png"/>
    <img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Technology Stack

ERPNext is built upon a robust and flexible technology stack:

*   **Frappe Framework:** A full-stack web application framework written in Python and Javascript, providing the foundation for the application, including database abstraction, user authentication, and a REST API. ([Frappe Framework GitHub](https://github.com/frappe/frappe))
*   **Frappe UI:** A Vue.js-based UI library that offers a modern and user-friendly interface, built on top of the Frappe Framework. ([Frappe UI GitHub](https://github.com/frappe/frappe-ui))

## Production Setup

### Managed Hosting

For a hassle-free experience, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. It handles installation, upgrades, monitoring, maintenance, and support.  It offers a fully featured developer platform with the ability to manage multiple Frappe deployments.

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

**Prerequisites:** Docker, Docker Compose, Git. For setup details, refer to the [Docker Documentation](https://docs.docker.com).

**Installation:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the Docker Compose command:
    ```bash
    docker compose -f pwd.yml up -d
    ```

Your site should be accessible on your localhost at port 8080 after a few minutes. Use the following credentials for initial login:
*   Username: `Administrator`
*   Password: `admin`

Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The easy way: Our install script for bench will install all dependencies (e.g. MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local Setup

Follow these steps to set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
    ```bash
    bench start
    ```
2.  In a separate terminal, run:
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
4.  Access the app in your browser at `http://erpnext.localhost:8000/app`.

## Learning and Community

*   [Frappe School](https://school.frappe.io): Learn ERPNext and the Frappe Framework through courses.
*   [Official documentation](https://docs.erpnext.com/): Extensive documentation.
*   [Discussion Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me): Get instant help from a large user community.

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