<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
    <h2>ERPNext: Open-Source ERP for Business Growth</h2>
    <p align="center">
        <p>Unleash the power of a fully featured ERP system, built for scalability and efficiency.</p>
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
    <br>
    <a href="https://github.com/frappe/erpnext">View on GitHub</a>
</div>

## About ERPNext

**ERPNext** is a comprehensive, 100% open-source Enterprise Resource Planning (ERP) system designed to streamline your business operations. It provides a powerful, intuitive platform to manage all aspects of your business, from accounting and sales to manufacturing and projects.  Manage your entire business from a single platform, and experience the freedom of open source.

### Key Features

*   **Accounting:**  Manage your finances with integrated tools for transactions, financial reporting, and cash flow management.
*   **Order Management:**  Track inventory, manage sales orders, handle customers and suppliers, and streamline order fulfillment.
*   **Manufacturing:**  Simplify the production cycle, track material consumption, plan capacity, and manage subcontracting.
*   **Asset Management:**  Track assets throughout their lifecycle, covering everything from IT infrastructure to equipment.
*   **Projects:**  Deliver both internal and external projects on time, within budget, and with a focus on profitability.

<details open>

<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

### Why Choose ERPNext?

ERPNext empowers you to run your business more efficiently and effectively by integrating all essential business functions into a single, easy-to-use platform.  Its open-source nature provides flexibility, customization options, and avoids vendor lock-in.

### Under the Hood

*   **Frappe Framework:**  The foundation of ERPNext, a full-stack web application framework written in Python and Javascript, providing a robust foundation for web applications.
*   **Frappe UI:** A Vue-based UI library to deliver a modern and intuitive user interface.

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for simplified ERPNext hosting.  Frappe Cloud handles installation, upgrades, monitoring, and maintenance.

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

**Prerequisites:** Docker, Docker Compose, Git.  Refer to [Docker Documentation](https://docs.docker.com) for setup details.

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run Docker Compose:

    ```bash
    docker compose -f pwd.yml up -d
    ```

    After a few minutes, your site will be accessible on `localhost:8080`.

3.  **Login:** Use the following default credentials:

    *   Username: `Administrator`
    *   Password: `admin`

    Refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based Docker setup.

## Development Setup

### Manual Install

The Easy Way: use our install script for bench which installs all dependencies (e.g. MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

To set up the repository locally, follow these steps:

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal, create a new site:

    ```bash
    bench new-site erpnext.localhost
    ```

3.  Get and install the ERPNext app:

    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```

4.  Access the app in your browser at `http://erpnext.localhost:8000/app`.

## Learning and Community

1.  [Frappe School](https://school.frappe.io): Learn Frappe Framework and ERPNext through courses.
2.  [Official documentation](https://docs.erpnext.com/): Comprehensive ERPNext documentation.
3.  [Discussion Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me): Get instant help from the user community.

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