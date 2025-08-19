<!-- Improved & Summarized README for ERPNext -->

<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Your Business</h2>
    <p align="center">
        <p><b>Revolutionize your business operations with ERPNext, the powerful, intuitive, and open-source Enterprise Resource Planning (ERP) system.</b></p>
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
    [View the Original Repository on GitHub](https://github.com/frappe/erpnext)
</div>

---

## What is ERPNext?

ERPNext is a 100% open-source ERP system designed to empower businesses of all sizes. It provides a comprehensive suite of tools to manage various aspects of your company, from accounting to manufacturing, all in one integrated platform.

## Key Features

*   **Accounting:** Manage your finances with ease, from transaction recording to comprehensive financial reporting.
*   **Order Management:** Streamline your sales process by tracking inventory, managing sales orders, handling customer and supplier interactions, and fulfilling orders efficiently.
*   **Manufacturing:** Simplify production cycles, track material consumption, manage capacity planning, and handle subcontracting.
*   **Asset Management:** Track assets throughout their lifecycle, from purchase to disposal, ensuring proper management of IT infrastructure and equipment.
*   **Projects:** Deliver projects on time and within budget by tracking tasks, timesheets, and issues, ensuring profitability.

<details open>
    <summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Under the Hood

ERPNext is built upon two core components:

*   **[Frappe Framework](https://github.com/frappe/frappe):** A full-stack web application framework, providing a robust foundation with features like database abstraction, user authentication, and a REST API.
*   **[Frappe UI](https://github.com/frappe/frappe-ui):** A modern Vue-based UI library that offers a range of components for building user-friendly applications on top of the Frappe Framework.

## Production Setup

### Managed Hosting

Simplify your ERPNext deployment with [Frappe Cloud](https://frappecloud.com). This platform offers a user-friendly solution for hosting Frappe applications, handling installation, upgrades, monitoring, and support.

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

For self-hosting with Docker, ensure you have Docker, docker-compose, and git installed.

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the Docker Compose file:
    ```bash
    docker compose -f pwd.yml up -d
    ```

Access your site on `localhost:8080` using the default credentials:

*   **Username:** Administrator
*   **Password:** admin

## Development Setup

### Manual Install

Follow these steps to set up ERPNext for development:

1.  Install bench: follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation).
2.  Start the server:
    ```bash
    bench start
    ```
3.  In a separate terminal, create a new site:
    ```bash
    bench new-site erpnext.localhost
    ```
4.  Get and install the ERPNext app:
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
5.  Open the app in your browser: `http://erpnext.localhost:8000/app`

## Learning and Community

*   **[Frappe School](https://school.frappe.io):** Learn ERPNext and the Frappe Framework through courses by maintainers and the community.
*   **[Official Documentation](https://docs.erpnext.com/):** Explore comprehensive documentation for ERPNext.
*   **[Discussion Forum](https://discuss.erpnext.com/):** Engage with the ERPNext community.
*   **[Telegram Group](https://erpnext_public.t.me):** Get instant help and connect with other users.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Review our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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