<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Growing Businesses</h2>
    <p align="center">
        <p>Empower your business with a powerful, intuitive, and open-source Enterprise Resource Planning (ERP) system.</p>
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

**ERPNext** is a 100% open-source ERP system designed to streamline your business operations, offering a comprehensive suite of modules to manage your business effectively. 

[**Explore the ERPNext Repository on GitHub**](https://github.com/frappe/erpnext)

### Key Features

*   **Accounting:** Manage your finances with tools for transactions, reports, and analysis.
*   **Order Management:** Control inventory, sales orders, customer relationships, and order fulfillment.
*   **Manufacturing:** Simplify production, track materials, and manage capacity planning.
*   **Asset Management:** Track assets throughout their lifecycle, from purchase to disposal.
*   **Projects:** Deliver projects on time, within budget, and with maximum profitability.

<details open>
<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Under the Hood

*   **Frappe Framework:** A robust, full-stack web application framework built with Python and JavaScript, providing a solid foundation for ERPNext.
*   **Frappe UI:** A modern Vue-based UI library that provides a clean and intuitive user experience.

## Deployment Options

### Managed Hosting

Experience the ease of use with [Frappe Cloud](https://frappecloud.com), a user-friendly platform to host Frappe applications. It handles all installation, setup, upgrades, and maintenance.

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

**Prerequisites:** `docker`, `docker-compose`, `git`. For Docker setup details, consult the [Docker Documentation](https://docs.docker.com).

**Steps to Run:**

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run Docker Compose:

    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your site will be accessible on `localhost:8080`. Use the following default login credentials:

*   Username: `Administrator`
*   Password: `admin`

For ARM-based Docker setups, refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Install

Install all dependencies using the provided install script for bench.  See [bench](https://github.com/frappe/bench) for more details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local Setup

To set up the repository locally, follow these steps:

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:

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

4.  Open the URL `http://erpnext.localhost:8000/app` in your browser to access the running app.

## Resources and Community

1.  [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext.
2.  [Official documentation](https://docs.erpnext.com/) - Detailed documentation for ERPNext.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from the user community.

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