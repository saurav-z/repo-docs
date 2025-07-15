<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Growing Businesses</h2>
    <p align="center">
        <b>Streamline your operations and empower your business with ERPNext, a powerful and intuitive open-source Enterprise Resource Planning (ERP) system.</b>
    </p>
    <p align="center">
      <a href="https://github.com/frappe/erpnext"><b>View on GitHub</b></a>
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

## What is ERPNext?

ERPNext is a 100% open-source ERP system designed to help businesses of all sizes manage their operations efficiently.  It offers a comprehensive suite of modules, from accounting and sales to manufacturing and project management, all within a single, integrated platform.

## Key Features of ERPNext

*   **Accounting:** Comprehensive tools to manage finances, from transactions to financial reporting.
*   **Order Management:** Efficiently track inventory, manage sales orders, and streamline order fulfillment.
*   **Manufacturing:** Simplify production cycles, track material consumption, and optimize capacity planning.
*   **Asset Management:** Manage your assets, from purchase to disposal, within a centralized system.
*   **Projects:** Deliver projects on time, within budget, and profitably, with task, timesheet, and issue tracking.
*   **Human Resource Management**: Manage your employee lifecycle from hire to retire.
*   **CRM (Customer Relationship Management)**: Manage customer relationships, track leads, and improve sales efficiency.

<details open>

<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Technology Stack

*   **Frappe Framework:** A full-stack web application framework built on Python and JavaScript, providing a robust foundation.
*   **Frappe UI:** A modern, Vue-based UI library that provides a responsive and user-friendly interface.

## Getting Started

### Production Setup

#### Managed Hosting (Recommended)

Experience the ease of hosting with [Frappe Cloud](https://frappecloud.com). This platform simplifies deployment, upgrades, and maintenance, allowing you to focus on your business.

<div>
	<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

#### Self-Hosted

##### Docker

1.  **Prerequisites:** Docker, docker-compose, and Git installed. See [Docker Documentation](https://docs.docker.com) for setup instructions.
2.  **Run:**
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    docker compose -f pwd.yml up -d
    ```
3.  Access your ERPNext instance at `localhost:8080`. Default login: Administrator / admin.

For ARM-based Docker setup, see [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

### Development Setup

#### Manual Install

**Using Bench (Recommended):** Our install script for bench will install all dependencies (e.g. MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

#### Local

1.  **Set up Bench:** Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
    ```bash
    bench start
    ```
2.  **Create a new site:** In a separate terminal:
    ```bash
    bench new-site erpnext.localhost
    ```
3.  **Get and install ERPNext:**
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
4.  **Access:** Open `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community Resources

1.  [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext through courses.
2.  [Official Documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from the user community.

## Contributing

We welcome contributions!  Please review our guidelines:

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

## License

ERPNext is licensed under the [MIT License](LICENSE).

## Trademark

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