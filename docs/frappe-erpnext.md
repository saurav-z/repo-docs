<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Businesses of All Sizes</h2>
    <p align="center">
        <p>Powering businesses with a comprehensive, intuitive, and open-source Enterprise Resource Planning (ERP) solution.</p>
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

**ERPNext is a fully open-source ERP system designed to streamline your business operations, offering a powerful and cost-effective alternative to proprietary solutions.**  [Explore the ERPNext repository on GitHub](https://github.com/frappe/erpnext).

### Key Features of ERPNext

*   **Accounting:** Manage finances with tools for transactions, financial reporting, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales orders, handle customer and supplier relationships, and streamline fulfillment.
*   **Manufacturing:** Simplify the production cycle, track material consumption, and support capacity planning and subcontracting.
*   **Asset Management:** Track your assets from purchase to disposal, covering all aspects of your organization.
*   **Projects:** Manage both internal and external projects, track tasks, and monitor time and expenses.

<details open>
<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Under the Hood

*   **[Frappe Framework](https://github.com/frappe/frappe):** A robust, full-stack web application framework (Python and Javascript).  It provides a foundation for building web applications.
*   **[Frappe UI](https://github.com/frappe/frappe-ui):** A modern, Vue-based UI library offering a suite of components for building user-friendly interfaces.

## Getting Started

### Production Setup

#### Managed Hosting

Simplify deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. It handles installation, upgrades, monitoring, and support.

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

**Prerequisites:** Docker, Docker Compose, Git.

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run the application:

    ```bash
    docker compose -f pwd.yml up -d
    ```

3.  Access your site at `localhost:8080`.
    *   **Default Login:**
        *   Username: `Administrator`
        *   Password: `admin`

    Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

### Development Setup

#### Manual Install

For a complete setup, use the following steps:

1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up `bench` and start the server.
    ```bash
    bench start
    ```

2.  Open a new terminal and run:
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

## Learning and Community Resources

1.  [Frappe School](https://school.frappe.io) - Learn from courses by the maintainers and community.
2.  [Official documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Connect with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from a large user base.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

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