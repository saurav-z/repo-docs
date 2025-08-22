<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
    <h2>ERPNext: Open-Source ERP Software for Businesses</h2>
    <p align="center">
        **Manage your entire business operations with ERPNext, a powerful, intuitive, and completely open-source ERP system.**
    </p>
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

---

## Overview

ERPNext is a 100% open-source Enterprise Resource Planning (ERP) system designed to streamline and improve business operations.  From accounting to manufacturing, ERPNext provides a comprehensive suite of modules to manage your business effectively.  This project is hosted on [GitHub](https://github.com/frappe/erpnext).

## Key Features

*   **Accounting:** Manage your finances with tools for transactions, financial reports, and cash flow management.
*   **Order Management:**  Track inventory, manage sales orders, and handle customer and supplier interactions seamlessly.
*   **Manufacturing:** Simplify your production cycle with tools for material consumption, capacity planning, and subcontracting.
*   **Asset Management:**  Track assets from purchase to disposal, across all areas of your organization.
*   **Projects:** Manage both internal and external projects on time, within budget, and profitably, tracking tasks and issues.

<details open>
    <summary>More</summary>
    <img src="https://erpnext.com/files/v16_bom.png"/>
    <img src="https://erpnext.com/files/v16_stock_summary.png"/>
    <img src="https://erpnext.com/files/v16_job_card.png"/>
    <img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Under the Hood

*   [**Frappe Framework**](https://github.com/frappe/frappe): The robust, full-stack web application framework built in Python and Javascript that powers ERPNext.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library providing a modern user interface.

## Installation and Setup

### Managed Hosting

Simplify your ERPNext deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.

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
2.  Run Docker Compose:
    ```bash
    docker compose -f pwd.yml up -d
    ```

Your ERPNext site will be accessible on `localhost:8080`. Use the following credentials to log in:

*   Username: Administrator
*   Password: admin

Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setups.

#### Manual Installation

For detailed instructions on manual installation, see the [Development Setup](#development-setup) section below.

## Development Setup

### Manual Install

1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up bench and start the server.
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

3.  Open `http://erpnext.localhost:8000/app` in your browser to access the application.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn ERPNext from official courses and community contributions.
*   [Official documentation](https://docs.erpnext.com/) - Comprehensive documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from the user community.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Please review the [Logo and Trademark Policy](TRADEMARK_POLICY.md).

---

<div align="center" style="padding-top: 0.75rem;">
	<a href="https://frappe.io" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png">
			<img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies" height="28"/>
		</picture>
	</a>
</div>