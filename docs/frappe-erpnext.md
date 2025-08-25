# ERPNext: Open-Source ERP for Growing Businesses

[![ERPNext Logo](erpnext/public/images/v16/erpnext.svg)](https://github.com/frappe/erpnext)

**ERPNext is a powerful, intuitive, and open-source Enterprise Resource Planning (ERP) system that empowers businesses to streamline operations and drive growth.**

[View the original repository on GitHub](https://github.com/frappe/erpnext)

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

[Live Demo](https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo) | [Website](https://frappe.io/erpnext) | [Documentation](https://docs.frappe.io/erpnext/)

## Key Features of ERPNext

ERPNext offers a comprehensive suite of modules to manage every aspect of your business:

*   **Accounting:** Manage your finances with tools for transactions, financial reporting, and cash flow analysis.
*   **Order Management:** Track inventory, manage sales orders, and fulfill orders efficiently, from customer to supplier.
*   **Manufacturing:** Streamline your production cycle, track material consumption, and manage subcontracting.
*   **Asset Management:** Track all your organization assets from purchase to perishment.
*   **Projects:** Manage internal and external projects on time, within budget, and for maximum profitability, including task and timesheet tracking.

<details open>
<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Under the Hood

ERPNext is built upon:

*   **Frappe Framework:** A full-stack web application framework written in Python and Javascript, providing a robust foundation for web application development. ([Frappe Framework on GitHub](https://github.com/frappe/frappe))
*   **Frappe UI:** A Vue-based UI library for a modern and user-friendly interface. ([Frappe UI on GitHub](https://github.com/frappe/frappe-ui))

## Production Setup

### Managed Hosting (Recommended)

[Frappe Cloud](https://frappecloud.com) offers a simple, user-friendly, and sophisticated platform to host Frappe applications. It takes care of installation, setup, upgrades, monitoring, maintenance, and support.

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

**Prerequisites:** Docker, Docker Compose, and Git. For Docker setup details, refer to the [Docker Documentation](https://docs.docker.com).

**Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run Docker Compose:
    ```bash
    docker compose -f pwd.yml up -d
    ```

After a couple of minutes, your site will be accessible on `localhost:8080`. Use the following credentials:

*   **Username:** Administrator
*   **Password:** admin

For ARM-based Docker setups, refer to the [Frappe Docker documentation](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Install

*   The Easy Way: Use our install script for bench, which installs all dependencies (e.g., MariaDB). More details can be found at: https://github.com/frappe/bench.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
    ```bash
    bench start
    ```
2.  In a separate terminal, run:
    ```bash
    # Create a new site
    bench new-site erpnext.localhost

    # Get the ERPNext app
    bench get-app https://github.com/frappe/erpnext

    # Install the app
    bench --site erpnext.localhost install-app erpnext
    ```
3.  Open `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext through courses from maintainers and the community.
*   [Official Documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help from a large user community.

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