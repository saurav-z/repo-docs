<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP Software for Growing Businesses</h2>
</div>

<p align="center">
  <a href="https://frappe.school"><img src="https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square" alt="Learn ERPNext on Frappe School"></a>
  <br>
  <a href="https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml"><img src="https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule" alt="CI Status"></a>
  <a href="https://hub.docker.com/r/frappe/erpnext-worker"><img src="https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg" alt="Docker Pulls"></a>
</p>

<div align="center">
	<img src="./erpnext/public/images/v16/hero_image.png" alt="ERPNext Hero Image"/>
</div>

<div align="center">
	<a href="https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo">Live Demo</a>
	-
	<a href="https://frappe.io/erpnext">Website</a>
	-
	<a href="https://docs.frappe.io/erpnext/">Documentation</a>
</div>

## About ERPNext

ERPNext is a powerful, intuitive, and **100% open-source** Enterprise Resource Planning (ERP) system, empowering businesses to streamline operations and drive growth. It offers a comprehensive suite of integrated modules, eliminating the need for separate software solutions and saving you time and money.  Get started today with this flexible, open-source solution! ([Back to original repository](https://github.com/frappe/erpnext))

### Key Features:

*   **Accounting:** Manage your finances with tools for transactions, financial reports, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales orders, suppliers, shipments, and fulfillment.
*   **Manufacturing:** Simplify the production cycle, track material usage, and manage capacity planning.
*   **Asset Management:** Track assets from purchase to disposal, covering all aspects of your organization's infrastructure.
*   **Projects:** Manage both internal and external projects, track tasks, timesheets, and profitability.

<details open>
<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png" alt="Bill of Materials Example"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary Example"/>
	<img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card Example"/>
	<img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks Example"/>
</details>

### Technology Under the Hood

*   **Frappe Framework:** The underlying full-stack web application framework (Python/JavaScript) providing the foundation for ERPNext, including a database abstraction layer, user authentication, and a REST API. ([Frappe Framework](https://github.com/frappe/frappe))
*   **Frappe UI:** A Vue-based UI library that provides a modern and user-friendly interface for the ERPNext application. ([Frappe UI](https://github.com/frappe/frappe-ui))

## Production Setup

### Managed Hosting

Experience hassle-free deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.

<a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
	<picture>
		<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
		<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
	</picture>
</a>

### Self-Hosted

#### Docker

**Prerequisites:** Docker, Docker Compose, Git. Refer to the [Docker Documentation](https://docs.docker.com) for setup instructions.

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run the Docker Compose command:

    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, access your site on `localhost:8080`.

*   **Default login:**
    *   Username: `Administrator`
    *   Password: `admin`

**Note:** See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The easiest way is via the install script for bench, which handles dependencies (e.g., MariaDB). See [bench](https://github.com/frappe/bench) for more details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local

Follow these steps to set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server.
    ```bash
    bench start
    ```

2.  In a separate terminal window, run the following commands:
    ```bash
    # Create a new site
    bench new-site erpnext.localhost
    ```

3.  Get the ERPNext app and install it:
    ```bash
    # Get the ERPNext app
    bench get-app https://github.com/frappe/erpnext

    # Install the app
    bench --site erpnext.localhost install-app erpnext
    ```

4.  Open the URL `http://erpnext.localhost:8000/app` in your browser to see the running app.

## Learning and Community

1.  [Frappe School](https://school.frappe.io) - Learn from courses by the maintainers and community.
2.  [Official documentation](https://docs.erpnext.com/) - Extensive documentation for ERPNext.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from the community.

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