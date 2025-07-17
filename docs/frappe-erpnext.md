<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Business Success</h2>
    <p align="center">
        **Transform your business with ERPNext, the powerful, intuitive, and 100% open-source ERP solution.**
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

## About ERPNext

ERPNext is a comprehensive, open-source Enterprise Resource Planning (ERP) system designed to streamline and integrate all aspects of your business operations.  Manage everything from accounting and inventory to manufacturing and customer relationships, all in one powerful platform.

### Key Features

*   **Accounting:** Manage your finances efficiently with tools for transaction recording, financial reporting, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales and purchase orders, handle shipments, and optimize order fulfillment.
*   **Manufacturing:** Simplify your production cycle with features for production planning, material consumption tracking, capacity planning, and subcontracting.
*   **Asset Management:** Track assets throughout their lifecycle, from purchase to disposal, for comprehensive management.
*   **Projects:** Manage projects effectively, track tasks, and monitor time and costs for both internal and external projects.

<details open>
<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Under the Hood

ERPNext is built on a robust technology stack:

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework written in Python and JavaScript, providing a solid foundation for building web applications with database abstraction, user authentication, and a REST API.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library, providing a modern and responsive user interface for a seamless user experience.

## Getting Started

### Production Setup

Choose the best way to deploy ERPNext for your needs.

#### Managed Hosting (Recommended)

For ease of use and peace of mind, consider [Frappe Cloud](https://frappecloud.com), a simple and sophisticated platform for hosting your Frappe applications. It handles installation, upgrades, monitoring, and maintenance, allowing you to focus on your business.

<div>
	<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

#### Self-Hosted

Follow the steps below to set up a self-hosted instance.

#### Docker

*Prerequisites: docker, docker-compose, git. Refer to the [Docker Documentation](https://docs.docker.com) for more details on Docker setup.*

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the Docker Compose configuration:
    ```bash
    docker compose -f pwd.yml up -d
    ```

Your ERPNext site should be accessible on your localhost port 8080 after a few minutes.
*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based docker setup.

## Development Setup

### Manual Install

Install ERPNext with the following steps:

1.  **Install Bench:** Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server.
    ```bash
    bench start
    ```

2.  **Create a Site:**
    ```bash
    bench new-site erpnext.localhost
    ```

3.  **Get and Install ERPNext:**
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```

4.  Open the URL `http://erpnext.localhost:8000/app` in your browser to access the running application.

## Learning and Community

Expand your knowledge and connect with the ERPNext community through these resources:

1.  [Frappe School](https://school.frappe.io) - Learn ERPNext through courses by the maintainers and community.
2.  [Official Documentation](https://docs.erpnext.com/) - Access comprehensive documentation for ERPNext.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext user and service provider community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from a large community of users.

## Contributing

Contribute to the ERPNext project and help improve it:

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Please review our [Logo and Trademark Policy](TRADEMARK_POLICY.md) for information about using the ERPNext logo and trademarks.

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