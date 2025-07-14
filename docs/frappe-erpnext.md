<div align="center">
    <a href="https://frappe.io/erpnext">
	<img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80xp"/>
    </a>
</div>

# ERPNext: Open-Source ERP Software for Growing Businesses

ERPNext is a powerful, intuitive, and 100% open-source Enterprise Resource Planning (ERP) system, offering a comprehensive solution to manage all aspects of your business.  [Explore the original repository](https://github.com/frappe/erpnext).

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

## Key Features of ERPNext

ERPNext offers a suite of integrated modules to streamline your business operations:

*   **Accounting:** Manage cash flow with tools for recording transactions, summarizing financial reports, and detailed analysis.
*   **Order Management:** Track inventory, manage sales orders, handle customer relationships, and streamline order fulfillment.
*   **Manufacturing:** Simplify your production cycle with features for tracking material consumption, capacity planning, and subcontracting.
*   **Asset Management:** From purchase to disposal, manage your assets efficiently across your entire organization.
*   **Projects:** Deliver projects on time and on budget with features for task management, timesheets, and issue tracking.

<details open>

<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>


## Under the Hood

ERPNext is built upon the following core components:

*   **Frappe Framework:** A robust full-stack web application framework (Python/Javascript) providing a solid foundation with a database abstraction layer, user authentication, and a REST API. [Frappe Framework](https://github.com/frappe/frappe)
*   **Frappe UI:** A modern Vue-based UI library for building intuitive user interfaces. [Frappe UI](https://github.com/frappe/frappe-ui)

## Production Setup

### Managed Hosting

Get started quickly with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. It handles installation, upgrades, monitoring, and maintenance.

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

Prerequisites: docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

Run the following commands:

```
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a couple of minutes, your site should be accessible on `localhost:8080`. Use the following default login credentials to access the site:

*   Username: `Administrator`
*   Password: `admin`

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The Easy Way: Use the install script for bench, which installs all dependencies (e.g., MariaDB). See [bench documentation](https://github.com/frappe/bench) for more details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local

To set up the repository locally, follow these steps:

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
    ```
    bench start
    ```

2.  In a separate terminal window, run the following commands:
    ```
    # Create a new site
    bench new-site erpnext.localhost
    ```

3.  Get the ERPNext app and install it:
    ```
    # Get the ERPNext app
    bench get-app https://github.com/frappe/erpnext

    # Install the app
    bench --site erpnext.localhost install-app erpnext
    ```

4.  Open the URL `http://erpnext.localhost:8000/app` in your browser, and you should see the app running.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext through courses.
*   [Official documentation](https://docs.erpnext.com/) - Comprehensive documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant help.

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