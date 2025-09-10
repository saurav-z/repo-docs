# ERPNext: Open-Source ERP for Business Management

**ERPNext is a powerful, intuitive, and open-source Enterprise Resource Planning (ERP) system designed to streamline your business operations.** ([See the original repository](https://github.com/frappe/erpnext))

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

[<img src="./erpnext/public/images/v16/hero_image.png" alt="ERPNext Hero Image" width="100%"/>](https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo)

**Key Features:**

*   **Accounting:** Manage your finances with comprehensive tools for transactions, reporting, and analysis.
*   **Order Management:** Track inventory, handle sales orders, manage customers and suppliers, and optimize order fulfillment.
*   **Manufacturing:** Simplify the production cycle with features for material tracking, capacity planning, and subcontracting.
*   **Asset Management:** Manage assets, from purchase to disposal, across your organization.
*   **Projects:** Manage projects on time, within budget and on target for profitability. Track tasks, timesheets, and issues.

<details open>
<summary>More</summary>
	<img src="https://erpnext.com/files/v16_bom.png"/>
	<img src="https://erpnext.com/files/v16_stock_summary.png"/>
	<img src="https://erpnext.com/files/v16_job_card.png"/>
	<img src="https://erpnext.com/files/v16_tasks.png"/>
</details>

## Core Technologies

*   **Frappe Framework:** A full-stack web application framework (Python and Javascript) providing a robust foundation.
*   **Frappe UI:** A Vue-based UI library for a modern user interface.

## Production Setup

### Managed Hosting

Experience the ease of Frappe Cloud, a user-friendly platform for hosting Frappe applications, handling installation, upgrades, and maintenance.

[<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />](https://erpnext-demo.frappe.cloud/app/home)

### Self-Hosted

#### Docker

Prerequisites: docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

Run following commands:

```
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a couple of minutes, site should be accessible on your localhost port: 8080. Use below default login credentials to access the site.
- Username: Administrator
- Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The Easy Way: our install script for bench will install all dependencies (e.g. MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the ERPNext "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

To setup the repository locally follow the steps mentioned below:

1. Setup bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server
   ```
   bench start
   ```

2. In a separate terminal window, run the following commands:
   ```
   # Create a new site
   bench new-site erpnext.localhost
   ```

3. Get the ERPNext app and install it
   ```
   # Get the ERPNext app
   bench get-app https://github.com/frappe/erpnext

   # Install the app
   bench --site erpnext.localhost install-app erpnext
   ```

4. Open the URL `http://erpnext.localhost:8000/app` in your browser, you should see the app running

## Learning and Community

1.  [Frappe School](https://school.frappe.io) - Learn Frappe Framework and ERPNext.
2.  [Official Documentation](https://docs.erpnext.com/) - Extensive ERPNext documentation.
3.  [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://erpnext_public.t.me) - Get instant help from the community.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Please read our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

<div align="center" style="padding-top: 0.75rem;">
	<a href="https://frappe.io" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png">
			<img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies" height="28"/>
		</picture>
	</a>
</div>