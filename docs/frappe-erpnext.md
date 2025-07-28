<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Growing Businesses</h2>
    <p align="center">
        <p>Empower your business with ERPNext, a powerful, intuitive, and completely open-source Enterprise Resource Planning (ERP) system.</p>
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
</div>

## About ERPNext

**ERPNext** is a comprehensive, 100% open-source ERP system designed to streamline your business operations.  Manage everything from accounting and inventory to manufacturing and project management, all in one integrated platform.

**[Explore the ERPNext Repository on GitHub](https://github.com/frappe/erpnext)**

### Key Features

*   **Accounting:** Manage your finances with tools for transaction recording, financial reporting, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales orders, handle customer and supplier relationships, and ensure efficient order fulfillment.
*   **Manufacturing:** Simplify production cycles, track material consumption, plan capacity, manage subcontracting, and optimize your manufacturing processes.
*   **Asset Management:** Oversee your organization's assets, from IT infrastructure to equipment, with comprehensive tracking from purchase to disposal.
*   **Projects:** Deliver projects on time, within budget, and with high profitability. Track tasks, manage timesheets, and address issues effectively.

<details open>
<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Under the Hood

*   **Frappe Framework:**  The robust Python and Javascript full-stack web application framework that powers ERPNext.
*   **Frappe UI:**  A Vue.js-based UI library providing a modern and intuitive user interface.

## Getting Started

### Production Setup

#### Managed Hosting

Experience the simplicity and sophistication of Frappe Cloud, a user-friendly platform for hosting Frappe applications. It handles installation, upgrades, monitoring, and maintenance, providing a fully featured developer platform.

<div>
	<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

#### Self-Hosted

**Docker:**

Prerequisites: docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for setup details.

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run Docker Compose:
    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, access your site on `localhost:8080` using the default login credentials:

*   Username: Administrator
*   Password: admin

For ARM-based Docker setups, see [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

### Development Setup

#### Manual Install

1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up bench and start the server:
    ```bash
    bench start
    ```

2.  Open a new terminal window and run the following commands:

    ```bash
    # Create a new site
    bench new-site erpnext.localhost
    ```

    ```bash
    # Get the ERPNext app
    bench get-app https://github.com/frappe/erpnext

    # Install the app
    bench --site erpnext.localhost install-app erpnext
    ```

3.  Access the app in your browser at `http://erpnext.localhost:8000/app`.

## Learning and Community

1.  **Frappe School:**  Learn ERPNext and the Frappe Framework through comprehensive courses.
2.  **Official Documentation:**  Explore detailed ERPNext documentation.
3.  **Discussion Forum:**  Connect with the ERPNext community.
4.  **Telegram Group:**  Get instant help from a large user community.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Review the [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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